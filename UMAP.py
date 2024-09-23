import argparse
import MDAnalysis as mda
from MDAnalysis.analysis import align
import numpy as np
import umap
from scipy.stats import gaussian_kde
from bokeh.plotting import figure, show, output_file
from bokeh.models import (
    ColumnDataSource, HoverTool, ColorBar, LinearColorMapper,
    Toggle, CustomJS, Button
)
from bokeh.transform import linear_cmap
from bokeh.palettes import Cividis256
from bokeh.layouts import column

def parse_arguments():
    parser = argparse.ArgumentParser(description="MD Simülasyon Verilerini UMAP ile Görselleştir ve Serbest Enerji Haritası Oluştur")
    parser.add_argument('pdb', type=str, help='PDB dosyasının yolu')
    parser.add_argument('xtc', type=str, help='XTC dosyasının yolu')
    parser.add_argument('-s', '--selection', type=str, default='protein',
                        help='Atom seçimi için ifade (varsayılan: "protein")')
    parser.add_argument('-o', '--output', type=str, default='umap_free_energy.html',
                        help='Çıktı HTML dosyasının adı (varsayılan: "umap_free_energy.html")')
    return parser.parse_args()

def load_universe(pdb_path, xtc_path):
    u = mda.Universe(pdb_path, xtc_path)
    return u

def select_atoms(universe, selection_string):
    selection = universe.select_atoms(selection_string)
    if len(selection) == 0:
        raise ValueError(f"'{selection_string}' seçimine uygun atom bulunamadı.")
    return selection

def align_to_reference(universe, selection_string):
    # İlk frame'i referans olarak kullanarak hizalama yapıyoruz
    align.AlignTraj(universe, universe, select=selection_string, in_memory=True).run()
    return universe

def extract_coordinates(universe, selection):
    num_frames = len(universe.trajectory)
    num_atoms = len(selection)
    coords = np.zeros((num_frames, num_atoms, 3))
    
    for i, ts in enumerate(universe.trajectory):
        coords[i] = selection.positions
    
    # Koordinatları düzleştir
    flattened = coords.reshape((num_frames, num_atoms * 3))
    return flattened

def perform_umap(data, n_neighbors=15, min_dist=0.1, n_components=2, random_state=13):
    reducer = umap.UMAP(n_neighbors=n_neighbors,
                        min_dist=min_dist,
                        n_components=n_components,
                        random_state=random_state)
    embedding = reducer.fit_transform(data)
    return embedding

def compute_free_energy(embedding, temperature=300):
    kB = 0.008314  # kJ/(mol·K)
    kde = gaussian_kde(embedding.T)
    density = kde(embedding.T)
    n0 = np.max(density)
    density = np.clip(density, a_min=1e-10, a_max=None)
    deltaG_over_kBT = -np.log(density / n0)
    return deltaG_over_kBT

def find_lowest_energy_position(embedding, deltaG):
    # En düşük enerji seviyesinin indeksi
    min_energy_index = np.argmin(deltaG)
    # En düşük enerji seviyesinin konumu
    lowest_energy_position = embedding[min_energy_index]
    return lowest_energy_position

def align_to_lowest_energy(embedding, lowest_energy_position):
    # Tüm noktaları en düşük enerji konumuna göre hizalar
    aligned_embedding = embedding - lowest_energy_position
    return aligned_embedding

def compute_kde_contour(embedding, N=1500):  # N değerini artırarak ızgarayı daha ince yapıyoruz
    x, y = embedding[:,0], embedding[:,1]
    
    # Sınırları %20 oranında genişletme
    padding_x = (x.max() - x.min()) * 0.2
    padding_y = (y.max() - y.min()) * 0.2

    xmin, xmax = x.min() - padding_x, x.max() + padding_x
    ymin, ymax = y.min() - padding_y, y.max() + padding_y
    
    # Daha geniş bir ızgara oluşturuyoruz
    X, Y = np.mgrid[xmin:xmax:N*1j, ymin:ymax:N*1j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([x, y])
    kernel = gaussian_kde(values)
    Z = np.reshape(kernel(positions).T, X.shape)
    
    return X, Y, Z

def create_bokeh_plot(embedding, deltaG, output_html):
    frame_indices = np.arange(len(embedding))

    # En düşük enerji seviyesinin konumunu bul
    lowest_energy_position = find_lowest_energy_position(embedding, deltaG)
    
    # Embedleri en düşük enerji konumuna göre hizala
    aligned_embedding = align_to_lowest_energy(embedding, lowest_energy_position)

    # Compute KDE for contour with aligned embedding
    X, Y, Z = compute_kde_contour(aligned_embedding, N=1000)
    
    density_palette = Cividis256[::-1]
    Z_min, Z_max = Z.min(), Z.max()
    density_mapper = LinearColorMapper(palette=density_palette, low=Z_min, high=Z_max)
    
    source = ColumnDataSource(data=dict(
        x=aligned_embedding[:,0],
        y=aligned_embedding[:,1],
        deltaG=deltaG,
        frame=frame_indices
    ))

    TOOLTIPS = [
        ("Frame", "@frame"),
        ("ΔG / (kB T)", "@deltaG{0.2f}"),
        ("X", "@x{0.2f}"),
        ("Y", "@y{0.2f}")
    ]

    deltaG_min = np.min(deltaG)
    deltaG_max = np.max(deltaG)
    free_energy_mapper = linear_cmap(field_name='deltaG', palette=Cividis256, low=deltaG_min, high=deltaG_max)

    p = figure(title="UMAP Reduction Results - Free Energy and Density Map",
               tools="pan,wheel_zoom,box_zoom,reset,hover,save,tap",
               tooltips=TOOLTIPS,
               width=1000, height=800,
               match_aspect=True,
               x_range=(X.min(), X.max()),
               y_range=(Y.min(), Y.max()))

    p.image(image=[Z], x=X.min(), y=Y.min(), dw=X.ptp(), dh=Y.ptp(),
            color_mapper=density_mapper, alpha=1)

    scatter = p.scatter('x', 'y', source=source, size=12, color=free_energy_mapper, 
                        alpha=0.8, selection_color="red", name="scatter", level='overlay')

    levels = np.linspace(np.min(Z), np.max(Z), 15)
    p.contour(x=X[:,0], y=Y[0,:], z=Z, levels=levels[1:], fill_color=density_palette, line_color=density_palette)

    color_bar = ColorBar(color_mapper=free_energy_mapper['transform'], 
                         label_standoff=12, 
                         width=8, 
                         location=(0,0),
                         title="ΔG / (kB T)")
    p.add_layout(color_bar, 'right')

    p.xaxis.axis_label = "UMAP Dimension 1"
    p.yaxis.axis_label = "UMAP Dimension 2"
    p.grid.grid_line_color = "black"
    p.grid.grid_line_alpha = 0.1

    callback = CustomJS(args=dict(source=source), code="""
        const indices = source.selected.indices;
        if (indices.length > 0) {
            const index = indices[0];
            const frame = source.data['frame'][index];
            alert('Selected Frame: ' + frame);
        }
    """)

    p.js_on_event('tap', callback)

    toggle = Toggle(label="Toggle Scatter Points", button_type="success")
    toggle.js_on_click(CustomJS(args=dict(scatter=scatter), code="""
        scatter.visible = !scatter.visible;
    """))

    layout = column(toggle, p)

    output_file(output_html)
    show(layout)


def main():
    args = parse_arguments()
    
    print("Universe yükleniyor...")
    u = load_universe(args.pdb, args.xtc)
    
    print(f"Atomlar seçiliyor: {args.selection}")
    selection = select_atoms(u, args.selection)
    
    print("Frame'ler referans frame'e hizalanıyor...")
    align_to_reference(u, args.selection)
    
    print("Koordinatlar çıkarılıyor...")
    data = extract_coordinates(u, selection)
    
    print("UMAP indirgeme yapılıyor...")
    embedding = perform_umap(data)
    
    print("Serbest enerji hesaplanıyor...")
    deltaG = compute_free_energy(embedding)
    
    print("Bokeh grafiği oluşturuluyor...")
    create_bokeh_plot(embedding, deltaG, args.output)
    print(f"Görselleştirme '{args.output}' dosyasına kaydedildi.")

if __name__ == "__main__":
    main()
