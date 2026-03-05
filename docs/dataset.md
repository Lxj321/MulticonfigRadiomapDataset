# Dataset

## Folder structure

```text
Dataset/
  beam_maps/
    <config_id>/
      u0/
        beam_XX_angle_*.npy
        u0_all_beams.npz
        beam_settings.txt
  configs/
    *.txt
  height_maps/
    u1..u800/
      u*_height_matrix.npy
      u*_height_matrix_coords.npz
  radiomaps/
    <config_id>_beamXX/
      u1..u800_labeled_radiomap.npy
      beam_settings.txt
  sionna_maps/ (optional)
    u1..u800/meshes/*.ply



**Indexing**

**Scene ID**: u1..u800

**Configuration ID**: freq_{f}GHz_{NTR}TR_{B}beams_pattern_tr38901

**Beam ID**: beam00..beam{max} (varies with configuration)

**Beam maps use u0**: configuration-only (environment-independent) beam map features

**Notes**

This page will be expanded with exact tensor shapes/units and coordinate conventions.
