import os
import argparse
import numpy as np
import nibabel as nib
from nibabel import processing
from skimage.measure import marching_cubes
import pyvista as pv

# -------------------- I/O helpers (kept from your version) --------------------


def load_canonical(path):
    img = nib.load(path)
    return nib.as_closest_canonical(img)  # works across nibabel versions


def resample_to(img, target, order=0):
    """Resample img to target grid (shape+affine). order=0 for masks."""
    if img.shape != target.shape or not np.allclose(img.affine, target.affine, atol=1e-3):
        img = processing.resample_from_to(
            img, (target.shape, target.affine), order=order)
    return img

# -------------------- Mesh extraction & rendering (PyVista) -------------------


def iso_surface_to_pv(data, affine, level=0.5, smooth_iters=10):
    # Marching cubes in voxel coords (z, y, x)
    v_zyx, faces, _, _ = marching_cubes(data.astype(np.float32), level=level)
    ijk = v_zyx[:, [2, 1, 0]]
    xyz = nib.affines.apply_affine(affine, ijk)  # world coords (mm)

    # VTK wants a leading "3" per triangle
    faces_vtk = np.hstack(
        [np.full((faces.shape[0], 1), 3), faces]).astype(np.int32).ravel()
    mesh = pv.PolyData(xyz, faces_vtk).triangulate()

    if smooth_iters > 0:
        mesh = mesh.smooth(n_iter=smooth_iters,
                           relaxation_factor=0.1, feature_smoothing=False)

    mesh = mesh.compute_normals(
        cell_normals=False,          # we only need smooth point normals
        point_normals=True,
        split_vertices=False,        # 0.46.0 name (not "splitting")
        auto_orient_normals=True,
        consistent_normals=True
    )

    return mesh


def camera_from_elev_azim(bounds_min, bounds_max, elev=20, azim=130, dist_scale=2.2):
    c = 0.5 * (bounds_min + bounds_max)
    diag = np.linalg.norm(bounds_max - bounds_min)
    dist = dist_scale * (diag + 1e-6)
    el, az = np.deg2rad(elev), np.deg2rad(azim)
    view = np.array([np.cos(el)*np.cos(az), np.cos(el)*np.sin(az), np.sin(el)])
    pos = c + dist * view
    return [pos.tolist(), c.tolist(), [0, 0, 1]]


def plotter_with_lights(bg="black"):
    p = pv.Plotter(off_screen=True, window_size=(1000, 1000), lighting="none")
    p.set_background(bg)
    p.enable_depth_peeling(number_of_peels=32, occlusion_ratio=0.0)
    p.enable_eye_dome_lighting()
    p.add_light(pv.Light(position=(1, 1, 1),  color='white', intensity=1.0))
    p.add_light(pv.Light(position=(-1, -0.5, 0.5),
                color='white', intensity=0.6))
    return p


def render_single(mesh, out_path, cam_pos, bg="black", color=(1, 0.2, 0.2), opacity=0.80):
    p = plotter_with_lights(bg)
    p.add_mesh(mesh, color=color, opacity=opacity, smooth_shading=True,
               specular=0.35, specular_power=20, ambient=0.25, diffuse=0.9)
    p.camera_position = cam_pos
    p.show(screenshot=out_path)


def render_overlay(mesh1, mesh2, out_path, cam_pos, bg="black",
                   color1=(0.08, 0.35, 0.35),     # inner (greenish)
                   color2=(0.82, 0.82, 0.86),     # outer (light gray)
                   inner_opacity=0.92,
                   outer_opacity=0.40,            # front faces of outer
                   outer_backface=0.65,           # interior faces: more opaque
                   outline_outer=True, wire=False):

    p = pv.Plotter(off_screen=True, window_size=(1000, 1000), lighting="none")
    p.set_background(bg)
    p.enable_depth_peeling(number_of_peels=32, occlusion_ratio=0.0)
    # turn OFF EDL here (it makes the inner edges “pop” forward through the shell)
    # p.enable_eye_dome_lighting()

    # lights
    p.add_light(pv.Light(position=(1, 1, 1), intensity=1.0))
    p.add_light(pv.Light(position=(-1, -0.5, 0.5), intensity=0.6))

    # 1) draw INNER first
    p.add_mesh(
        mesh1, color=color1, opacity=inner_opacity, smooth_shading=True,
        specular=0.35, specular_power=20, ambient=0.25, diffuse=0.9
    )

    # 2) draw OUTER second (front faces slightly translucent, backfaces darker & less translucent)
    p.add_mesh(
        mesh2, color=color2, opacity=outer_opacity, smooth_shading=True,
        backface_params=dict(
            color=color2, opacity=min(1.0, outer_opacity+0.1)),
        show_edges=False
    )

    if wire:
        p.add_mesh(mesh2, style="wireframe", color="black",
                   opacity=0.5, line_width=0.6)

    p.camera_position = cam_pos
    p.show(screenshot=out_path)


# -------------------- Main with your path layout ------------------------------
def main(args):
    experiment = args.experiment
    patient = args.patient
    sourcetype = args.type
    data_dir = os.path.join(
        "examples", "data", "PatienTumorMultiScan2024", patient)
    res_dir = os.path.join(
        "examples", "patient-reconstruct", "results",
        f"simulation_{experiment}", patient)
    out_dir = os.path.join(
        "examples", "patient-reconstruct", "results",
        f"img3d_{experiment}", patient)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if experiment == "multiscan":
        idx1, idx2 = 1, 2
    elif experiment == "fixinit":
        idx1, idx2 = 2, 3
    else:
        raise ValueError("Unrecognized experiment type.")

    if sourcetype == "original":
        tumor1_path = os.path.join(
            data_dir, f"{patient}{idx1}_t1mask_resized.nii.gz")
        tumor2_path = os.path.join(
            data_dir, f"{patient}{idx2}_t1mask_resized.nii.gz")
    elif sourcetype == "simulation":
        # 0.3 is used for tumor contour plots
        args.thr1 = 0.1
        args.thr2 = 0.3
        if experiment == "multiscan":
            raise ValueError("Not implemented.")
        tumor1_path = os.path.join(
            res_dir, f"{patient}-i0.nii.gz")
        tumor2_path = os.path.join(
            res_dir, f"{patient}-i1000.nii.gz")
    else:
        raise ValueError("Unrecognized source type.")

    ref_img = load_canonical(os.path.join(
            data_dir, f"{patient}{idx1}_t1mask_resized.nii.gz"))
    img1 = resample_to(load_canonical(tumor1_path), ref_img, order=0)
    img2 = resample_to(load_canonical(tumor2_path), img1, order=0)  # nearest NN for masks

    print("Threshold: ", args.thr1)

    m1 = iso_surface_to_pv(img1.get_fdata(), img1.affine,
                           level=args.thr1, smooth_iters=args.smooth)
    m2 = iso_surface_to_pv(img2.get_fdata(), img2.affine,
                           level=args.thr2, smooth_iters=args.smooth)

    print("inner bounds:", m1.bounds)
    print("outer bounds:", m2.bounds)

    # Global camera from combined bounds
    bmin1 = np.array([m1.bounds[0], m1.bounds[2], m1.bounds[4]])
    bmax1 = np.array([m1.bounds[1], m1.bounds[3], m1.bounds[5]])
    bmin2 = np.array([m2.bounds[0], m2.bounds[2], m2.bounds[4]])
    bmax2 = np.array([m2.bounds[1], m2.bounds[3], m2.bounds[5]])
    bmin = np.minimum(bmin1, bmin2)
    bmax = np.maximum(bmax1, bmax2)
    cam = camera_from_elev_azim(
        bmin, bmax, args.elev, args.azim, dist_scale=args.dist)

    # Output paths in your data dir
    out1 = os.path.join(out_dir, args.out1)
    out2 = os.path.join(out_dir, args.out2)
    out_overlay = os.path.join(out_dir, args.out_overlay)

    render_single(m1, f"{out1}-{sourcetype}.png", cam,
        bg=args.bg, color=(1, 0.2, 0.2), opacity=args.single_opacity)
    render_single(m2, f"{out2}-{sourcetype}.png", cam,
        bg=args.bg, color=(0.8, 0.82, 0.88), opacity=args.single_opacity)
    render_overlay(
        m1, m2, f"{out_overlay}-{sourcetype}.png", cam, bg=args.bg,
        color1=(1, 0.2, 0.2), color2=(0.8, 0.82, 0.88),
        inner_opacity=args.inner_opacity,
        outer_opacity=args.outer_opacity,
        outer_backface=args.outer_backface,
        outline_outer=not args.no_outline,
        wire=args.wire
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="3D tumor surfaces + overlay (PyVista/VTK)")
    ap.add_argument('--patient', type=str, default="NIU-HUI")
    ap.add_argument('--experiment', type=str,
                    choices=['fixinit', 'multiscan'], default="fixinit")
    ap.add_argument('--type', type=str, default="simulation",
                    choices=["original", "simulation"])
    ap.add_argument("--thr1", type=float, default=0.5)
    ap.add_argument("--thr2", type=float, default=0.5)
    ap.add_argument("--smooth", type=int, default=10,
                    help="surface smoothing iterations")
    ap.add_argument("--elev", type=float, default=20.0)
    ap.add_argument("--azim", type=float, default=130.0)
    ap.add_argument("--dist", type=float, default=2.2,
                    help="camera distance scale")
    ap.add_argument("--bg", default="black")
    ap.add_argument("--single_opacity", type=float,
                    default=0.80, help="opacity for single renders")
    ap.add_argument("--inner_opacity",  type=float,
                    default=0.90, help="overlay: inner mesh opacity")
    ap.add_argument("--outer_opacity",  type=float,
                    default=0.35, help="overlay: outer mesh opacity")
    ap.add_argument("--outer_backface", type=float, default=0.20,
                    help="overlay: interior-face opacity for outer mesh")
    ap.add_argument("--wire", action="store_true",
                    help="overlay: add thin wireframe on outer mesh")
    ap.add_argument("--out1", default="tumor1")
    ap.add_argument("--out2", default="tumor2")
    ap.add_argument("--out_overlay", default="overlay")
    ap.add_argument("--no_outline", action="store_true",
                    help="disable silhouette on outer mesh")
    args = ap.parse_args()
    main(args)
