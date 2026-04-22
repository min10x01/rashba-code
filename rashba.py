from functools import lru_cache
import numpy as np
import matplotlib.pyplot as plt

def merge_band_lists(band_lists):
    """
    Merge multiple band data lists by averaging the energies and occupancies.
    Assumes all band data lists are lists of dictionaries with keys 'band', 'energy', 'occ'.
    """
    if not band_lists:
        return []
    num_bands = len(band_lists[0])
    merged = []
    for b in range(num_bands):
        sum_energy = 0.0
        sum_occ = 0.0
        for band_list in band_lists:
            sum_energy += band_list[b]['energy']
            sum_occ += band_list[b]['occ']
        merged.append({
            'band': band_lists[0][b]['band'],
            'energy': sum_energy / len(band_lists),
            'occ': sum_occ / len(band_lists)
        })
    return merged

def parse_outcar(file_path, hs_points, hs_labels):
    """
    Refined parser for OUTCAR files.
    """
    fermi_energy = None
    kpoints = []
    bands_data = []
    current_kpoint_bands = []
    reading_bands = False

    try:
        with open(file_path, 'r') as f:
            for line in f:
                line_stripped = line.strip()

                if "E-fermi" in line_stripped and fermi_energy is None:
                    parts = line_stripped.split()
                    try: 
                        fermi_energy = float(parts[2])
                    except (IndexError, ValueError):
                        fermi_energy = None
                    continue

                if "k-point" in line_stripped.lower():
                    if current_kpoint_bands:
                        bands_data.append(current_kpoint_bands)
                        current_kpoint_bands = []

                    if ':' in line_stripped:
                        parts_split = line_stripped.split(':', 1)
                        coords_part = parts_split[1].split()
                        if len(coords_part) == 3:
                            try:
                                kx, ky, kz = map(float, coords_part)
                                kpoints.append((kx, ky, kz))
                            except ValueError:
                                pass
                    continue

                if "band No." in line_stripped:
                    reading_bands = True
                    continue

                if reading_bands:
                    if "-------------------------------" in line_stripped:
                        reading_bands = False
                        continue

                    parts = line_stripped.split()
                    if len(parts) == 3:
                        try:
                            band_index = int(parts[0])
                            band_energy = float(parts[1])
                            band_occ = float(parts[2])
                            current_kpoint_bands.append({
                                'band': band_index,
                                'energy': band_energy,
                                'occ': band_occ
                            })
                        except ValueError:
                            pass

            if current_kpoint_bands:
                bands_data.append(current_kpoint_bands)

    except FileNotFoundError:
        print(f"File not found: {file_path}")

    merged_kpoints = []
    merged_bands_data = []
    i = 0
    
    while i < len(kpoints):
        pt = kpoints[i]
        if pt in hs_points:
            block = []
            while i < len(kpoints) and kpoints[i] == pt:
                block.append(bands_data[i])
                i += 1
            merged_kpoints.append(pt)
            merged_bands_data.append(merge_band_lists(block))
        else:
            merged_kpoints.append(pt)
            merged_bands_data.append(bands_data[i])
            i += 1

    return fermi_energy, merged_kpoints, merged_bands_data

def parse_kpoints(file_path):
    """
    Parse the KPOINTS file to extract high-symmetry k-points with labels.
    Supports two formats:
      - '...' lines ending with '! LABEL'
      - columns with explicit label in fourth column
    """
    high_symmetry_points = []
    labels = []
    k_density = None
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            if len(lines) >= 2:
                parts = lines[1].split()
                try:
                    k_density = int(parts[0])
                except ValueError:
                    k_density = None
            # In line-mode KPOINTS the first four lines are header metadata.
            for line in lines[4:]:
                stripped = line.strip()
                if not stripped:
                    continue
                if '!' in line:
                    coord_part, comment = line.split('!', 1)
                    parts = coord_part.split()
                    label = comment.strip().split()[0]
                else:
                    parts = stripped.split()
                    label = None
                    if len(parts) >= 4:
                        label = parts[3]
                if len(parts) < 3:
                    continue
                try:
                    x, y, z = map(float, parts[:3])
                except ValueError:
                    continue
                if label is None:
                    label = ''
                high_symmetry_points.append((x, y, z))
                labels.append(label)
    except FileNotFoundError:
        print(f"File not found: {file_path}")

    return high_symmetry_points, labels, k_density

@lru_cache(maxsize=1)
def extract_reciprocal_lattice(outcar_file):
    with open(outcar_file, "r") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if "reciprocal lattice vectors" in line:
            b1 = np.array([float(x) for x in lines[i+1].split()[3:6]])
            b2 = np.array([float(x) for x in lines[i+2].split()[3:6]])
            b3 = np.array([float(x) for x in lines[i+3].split()[3:6]])
            return b1, b2, b3

    raise ValueError(f"Could not find reciprocal lattice vectors in {outcar_file}.")

def validate_parsed_data(fermi_energy, kpts, bands_data, k_density):
    if fermi_energy is None:
        raise ValueError("Could not parse the Fermi energy from OUTCAR.")
    if not kpts:
        raise ValueError("No k-points were parsed from OUTCAR.")
    if not bands_data:
        raise ValueError("No band energies were parsed from OUTCAR.")
    if len(kpts) != len(bands_data):
        raise ValueError(
            f"Parsed {len(kpts)} k-points but {len(bands_data)} band blocks from OUTCAR."
        )
    if k_density is None or k_density < 1:
        raise ValueError("Could not parse a valid k-point line density from KPOINTS.")

def build_band_arrays(kpoints, bands_data):
    """
    Convert bands_data (list-of-lists) into two NumPy arrays:
      energies[k, b], occupancies[k, b]

    Assumes each k-point has the same max band index.
    """
    nk = len(kpoints)
    if nk == 0:
        raise ValueError("No k-points available to build band arrays.")
    max_band_index = max(bd['band'] for k_bands in bands_data for bd in k_bands)

    energies = np.full((nk, max_band_index), np.nan)
    occupancies = np.full((nk, max_band_index), np.nan)

    for k_idx, k_bands in enumerate(bands_data):
        for bd in k_bands:
            b_idx = bd['band'] - 1
            energies[k_idx, b_idx] = bd['energy']
            occupancies[k_idx, b_idx] = bd['occ']

    return energies, occupancies

def find_vbm_cbm(energies, fermi_energy, kpoints):
    """
    Identify the valence band maximum (VBM) and conduction band minimum (CBM)
    based on their relation to the Fermi energy.
    """
    below_fermi = np.isfinite(energies) & (energies < fermi_energy)
    above_fermi = np.isfinite(energies) & (energies > fermi_energy)

    if not np.any(below_fermi) or not np.any(above_fermi):
        raise ValueError("Could not identify both valence and conduction states around the Fermi energy.")

    valence_energies = np.where(below_fermi, energies, -np.inf)
    conduction_energies = np.where(above_fermi, energies, np.inf)

    vbm_flat = int(np.argmax(valence_energies))
    cbm_flat = int(np.argmin(conduction_energies))

    vbm = float(valence_energies.flat[vbm_flat])
    cbm = float(conduction_energies.flat[cbm_flat])
    vbm_kpoint, vbm_band = np.unravel_index(vbm_flat, energies.shape)
    cbm_kpoint, cbm_band = np.unravel_index(cbm_flat, energies.shape)

    return vbm, cbm, vbm_kpoint, cbm_kpoint, vbm_band, cbm_band

def rashba(i, j, dE):
    k=i
    l=j
    b1, b2, b3 = extract_reciprocal_lattice("OUTCAR")
    
    k_cart = (k[0] * b1 + k[1] * b2 + k[2] * b3)
    l_cart = (l[0] * b1 + l[1] * b2 + l[2] * b3)
    dk = 2*np.pi*np.linalg.norm(k_cart - l_cart)
    alpha = 2*dE/(dk)
    return alpha

def find_tick_indices(kpts, hs_points, tol=1e-5):
    """
    For each high-symmetry point (from the KPOINTS file) find the index in merged_kpts
    that matches its coordinates (within a tolerance).
    """
    used = set()
    tick_indices = []

    if not hs_points:
        return []

    for i, gp in enumerate(hs_points):
        if i > 0 and np.allclose(gp, hs_points[i-1], atol=tol):
            tick_indices.append(tick_indices[-1])
            continue

        matches = [
            idx for idx, k in enumerate(kpts)
            if idx not in used and np.allclose(k, gp, atol=tol)
        ]

        if matches:
            idx = matches[0]
        else:
            unused = [idx for idx in range(len(kpts)) if idx not in used]
            distances = [
                np.linalg.norm(np.array(kpts[idx]) - np.array(gp))
                for idx in unused
            ]
            idx = unused[int(np.argmin(distances))]

        tick_indices.append(idx)
        used.add(idx)

    tick_ind = [tick_indices[0]]
    for i in range(1, len(tick_indices) - 1, 2):
        curr = tick_indices[i]
        nxt = tick_indices[i + 1]
        avg = (curr + nxt) / 2
        if curr == nxt:
            tick_ind.append(curr)
        else:
            tick_ind.append(avg)
    if len(tick_indices) > 1:
        tick_ind.append(tick_indices[-1])

    return tick_ind


def find_ticks(hs_labels):
    if not hs_labels:
        return []
    tick_labels = [hs_labels[0]]
    for i in range(1, len(hs_labels) - 1, 2):
        curr = hs_labels[i]
        nxt  = hs_labels[i + 1]

        if curr == nxt:
            tick_labels.append(curr)
        else:
            tick_labels.append(f"{curr}|{nxt}")
    tick_labels.append(hs_labels[len(hs_labels)-1])
    return tick_labels          

def system_type_check(fermi_energy, energies, kpts):
    finite_energies = np.where(np.isfinite(energies), energies, np.nan)
    valid_bands = ~np.all(np.isnan(finite_energies), axis=0)
    band_mins = np.nanmin(finite_energies[:, valid_bands], axis=0)
    band_maxs = np.nanmax(finite_energies[:, valid_bands], axis=0)
    crosses_fermi = (band_mins < fermi_energy) & (band_maxs > fermi_energy)
    if np.any(crosses_fermi):
        return 0.0

    vbm, cbm, *_ = find_vbm_cbm(energies, fermi_energy, kpts)
    return cbm - vbm

def describe_k_segment(point_idx, tick_ind, tick_labels):
    if not tick_ind or not tick_labels:
        return f"k-index {point_idx}"

    if len(tick_ind) == 1:
        return tick_labels[0]

    tick_positions = np.asarray(tick_ind, dtype=float)
    insert_idx = int(np.searchsorted(tick_positions, point_idx, side='right'))

    if insert_idx <= 0:
        return tick_labels[0]
    if insert_idx >= len(tick_labels):
        return f"{tick_labels[-2]}->{tick_labels[-1]}"
    return f"{tick_labels[insert_idx - 1]}->{tick_labels[insert_idx]}"

def plot_band(fermi_energy, energies, kpts, hs_points, hs_labels,k_density):
    
    nk, nb = energies.shape
    E_b=system_type_check(fermi_energy,energies,kpts)
    
    #------------------------------METALLIC SYSTEMS-------------------------------#
    if E_b == 0:
        print("Skipping VBM/CBM and Rashba analysis: system is metallic.")
        
        xvals = np.arange(nk)
        plt.figure(figsize=(8, 6))
        for b in range(nb):
            plt.plot(xvals, energies[:, b], 'blue', linewidth=0.8,label='Band Structure' if b == 0 else "")

        if fermi_energy is not None:
            plt.axhline(y=fermi_energy, color='limegreen', linestyle='--', label='Fermi Energy')
    
    #------------------------------NON-METALLIC SYSTEMS-------------------------------#        
    else:
        xvals = np.arange(nk)
        plt.figure(figsize=(8, 6))
        for b in range(nb):
            if np.all(energies[:, b] < fermi_energy):  # Valence bands
                plt.plot(xvals, energies[:, b], 'blue', linewidth=0.8,
                        label='Valence Band' if b == 0 else "")
            elif np.all(energies[:, b] > fermi_energy):  # Conduction bands
                plt.plot(xvals, energies[:, b], 'deeppink', linewidth=0.8,
                        label='Conduction Band' if b == 0 else "")
            else:
                plt.plot(xvals, energies[:, b], 'gray', linewidth=0.8,
                        label='Crossing Band' if b == 0 else "")

        if fermi_energy is not None:
            plt.axhline(y=fermi_energy, color='limegreen', linestyle='--', label='Fermi Energy')

        vbm, cbm, vbm_kpoint, cbm_kpoint, vbm_band, cbm_band = find_vbm_cbm(energies, fermi_energy, kpts)
        vb_minima, cb_minima = find_extrema(fermi_energy, energies,kpts,k_density)
        
        plt.scatter(xvals[vbm_kpoint], vbm, color='cyan', label='VBM', zorder=5, marker='o', s=100)
        plt.scatter(xvals[cbm_kpoint], cbm, color='red', label='CBM', zorder=5, marker='o', s=100)
        if len(cb_minima)==2:
            cb_minima_1=cb_minima[0]
            cb_minima_2=cb_minima[1]
            plt.scatter(xvals[cb_minima_2], energies[cb_minima_2, cbm_band], color='red', zorder=5, marker='o', s=100)
        if len(vb_minima)==2:
            vb_minima_1=vb_minima[0]
            vb_minima_2=vb_minima[1]
            plt.scatter(xvals[vb_minima_2], energies[vb_minima_2, vbm_band], color='cyan', 
                        zorder=5, marker='o', s=100)
            
    tick_ind = find_tick_indices(kpts, hs_points)
    tick_labels= find_ticks(hs_labels)
    
    plt.xticks(tick_ind, tick_labels)
    for xt in tick_ind:
        plt.axvline(x=xt, color='black', linestyle='-', linewidth=1)

    plt.xlim([0, nk - 1])
    plt.ylim(fermi_energy - 3, fermi_energy + 3)
    plt.xlabel("k-point index")
    plt.ylabel("Energy (eV)")
    plt.title("Band Structure from OUTCAR")
    #plt.legend()
    plt.show()
    
def find_extrema(fermi_energy, energies, kpts, k_density):
    
    vbm, cbm, vbm_kpoint, cbm_kpoint, vbm_band, cbm_band = find_vbm_cbm(energies, fermi_energy, kpts)

    kpt_range = max(1, int(k_density/2))
    kpt_index = cbm_kpoint
    cb_minima_2 = None
    for i in range(-kpt_range, kpt_range + 1):
        if i == 0:
            continue
        idx = kpt_index + i
        if idx - 1 < 0 or idx + 1 >= energies.shape[0]:
            continue
        if (energies[idx, cbm_band] <= energies[idx + 1, cbm_band] and 
            energies[idx, cbm_band] <= energies[idx - 1, cbm_band]):
            cb_minima_2 = idx
    cb_minima_1 = cbm_kpoint
    cb_minima=[cb_minima_1]
    if cb_minima_2 is not None:
        cb_minima.append(cb_minima_2)
    else:
        cb_minima = [cb_minima_1]
    
    # For Valence Band (VB)
    kpt_index = vbm_kpoint
    vb_minima_2 = None
    for i in range(-kpt_range, kpt_range + 1):
        if i == 0:
            continue
        idx = kpt_index + i
        if idx - 1 < 0 or idx + 1 >= energies.shape[0]:
            continue
        if (energies[idx, vbm_band] > energies[idx + 1, vbm_band] and 
            energies[idx, vbm_band] > energies[idx - 1, vbm_band]):
            vb_minima_2 = idx
    vb_minima_1 = vbm_kpoint
    vb_minima=[vb_minima_1]
    if vb_minima_2 is not None:
        vb_minima.append(vb_minima_2)
    else:
        vb_minima = [vb_minima_1]
    
    return vb_minima, cb_minima  

def find_split(fermi_energy, energies, kpts, hs_points, hs_labels, k_density):

    tol = 1e-4 
    vbm, cbm, vbm_kpoint, cbm_kpoint, vbm_band, cbm_band = find_vbm_cbm(energies, fermi_energy, kpts)
    vb_minima, cb_minima = find_extrema(fermi_energy, energies, kpts, k_density)
    
    tick_ind = find_tick_indices(kpts, hs_points)
    tick_labels= find_ticks(hs_labels)
    
    if len(cb_minima) == 1:
        cb_start_idx = cb_minima[0]
        k=int(k_density/2)
        start = max(0, cb_start_idx - k)
        end = min(energies.shape[0] - 1, cb_start_idx + k)
        if cbm_band + 1 >= energies.shape[1]:
            print("No Rashba in CB")
        else:
            band0 = energies[start:end+1, cbm_band]
            band1 = energies[start:end+1, cbm_band+1]
            good = np.isclose(band0, band1, atol=tol)

            if np.any(good):
                masked = band0.copy()
                masked[~good] = -np.inf

                offset = int(np.argmax(masked))
                cb_int_pt = start + offset
            else:
                cb_int_pt = None

            if cb_int_pt is not None and np.isclose(energies[cb_int_pt, cbm_band],energies[cb_int_pt, cbm_band+1],atol=tol, rtol=0):
                print(f"Conduction band splitting point is at k-point index {cb_int_pt}({kpts[cb_int_pt]})")
                
                dE = energies[cb_int_pt, cbm_band] - energies[cb_minima[0], cbm_band]
                cb_alpha = rashba(kpts[cb_int_pt],kpts[cb_minima[0]],dE)
                print(f"α ({describe_k_segment(cb_int_pt, tick_ind, tick_labels)}) on CB: {cb_alpha:.3f}")
            else:
                print("No Rashba in CB")

        
    elif len(cb_minima) == 2:
        cb_start_idx, cb_end_idx = sorted(cb_minima)
        if cbm_band + 1 >= energies.shape[1]:
            print("No Rashba in CB")
        else:
            conduction_range = energies[cb_start_idx:cb_end_idx+1, cbm_band]

            offset = np.argmax(conduction_range)
            cb_int_pt = cb_start_idx + offset
            if np.isclose(energies[cb_int_pt, cbm_band],energies[cb_int_pt, cbm_band+1],atol=tol, rtol=0):
                print(f"Conduction band splitting point is at k-point index {cb_int_pt}({kpts[cb_int_pt]})")
                dE1 = energies[cb_int_pt, cbm_band] - energies[cb_minima[0], cbm_band]
                dE2 = energies[cb_int_pt, cbm_band] - energies[cb_minima[1], cbm_band]
                cb_alpha_1 = rashba(kpts[cb_int_pt],kpts[cb_minima[0]],dE1)
                cb_alpha_2 = rashba(kpts[cb_int_pt],kpts[cb_minima[1]],dE2)
                segment = describe_k_segment(cb_int_pt, tick_ind, tick_labels)
                print(f"α ({segment}) on CB: {cb_alpha_1:.3f}")
                print(f"α ({segment}) on CB: {cb_alpha_2:.3f}")
            else:
                print("No Rashba in CB")
    else:
        print("No Rashba in CB")

    print()

    if len(vb_minima) == 1:
        vb_start_idx = vb_minima[0]
        k=int(k_density/2)
        start = max(0, vb_start_idx - k)
        end = min(energies.shape[0] - 1, vb_start_idx + k)
        if vbm_band - 1 < 0:
            print("No Rashba in VB")
        else:
            band0 = energies[start:end+1, vbm_band]
            band1 = energies[start:end+1, vbm_band-1]
            good = np.isclose(band0, band1, atol=tol)

            if np.any(good):
                masked = band0.copy()
                masked[~good] = np.inf

                offset = int(np.argmin(masked))
                vb_int_pt = start + offset
            else:
                vb_int_pt = None

            if vb_int_pt is not None and np.isclose(energies[vb_int_pt, vbm_band],energies[vb_int_pt, vbm_band-1],atol=tol, rtol=0):
                print(f"Valence band splitting point is at k-point index {vb_int_pt}({kpts[vb_int_pt]})")
                
                dE = energies[vb_minima[0], vbm_band] - energies[vb_int_pt, vbm_band]
                vb_alpha = rashba(kpts[vb_int_pt],kpts[vb_minima[0]],dE)
                print(f"α ({describe_k_segment(vb_int_pt, tick_ind, tick_labels)}) on VB: {vb_alpha:.3f}")
            else:
                print("No Rashba in VB")
            
            
    elif len(vb_minima) == 2:
        vb_start_idx, vb_end_idx = sorted(vb_minima)
        if vbm_band - 1 < 0:
            print("No Rashba in VB")
        else:
            valence_range = energies[vb_start_idx:vb_end_idx+1, vbm_band]

            offset = np.argmin(valence_range)
            vb_int_pt = vb_start_idx + offset
            if np.isclose(energies[vb_int_pt, vbm_band],energies[vb_int_pt, vbm_band-1],atol=tol, rtol=0):
                print(f"Valence band splitting point is at k-point index {vb_int_pt}({kpts[vb_int_pt]})")
                dE1 = energies[vb_minima[0], vbm_band] - energies[vb_int_pt, vbm_band]
                vb_alpha_1 = rashba(kpts[vb_int_pt],kpts[vb_minima[0]],dE1)
                dE2 = energies[vb_minima[1], vbm_band] - energies[vb_int_pt, vbm_band]
                vb_alpha_2 = rashba(kpts[vb_int_pt],kpts[vb_minima[1]],dE2)
                segment = describe_k_segment(vb_int_pt, tick_ind, tick_labels)
                print(f"α ({segment}) on VB: {vb_alpha_1:.3f}")
                print(f"α ({segment}) on VB: {vb_alpha_2:.3f}")
            else:
                print("No Rashba in VB")
    else:
        print("No Rashba in VB")


def main():
    print("""**************************************************************************************************
      
                                     RASHBA PARAMETER Code                                           
                                                                                                 
  By Sankalpa, Taranga & Ayushi
  
  #Run the code in a VASP band structure calculation directory with OUTCAR and KPOINTS
**************************************************************************************************""")


    outcar_file = "OUTCAR"
    kpoints_file = "KPOINTS"

    hs_points, hs_labels, k_density = parse_kpoints(kpoints_file)

    fermi_energy, kpts, bands_data = parse_outcar(outcar_file, hs_points, hs_labels)
    validate_parsed_data(fermi_energy, kpts, bands_data, k_density)
    print(f"Fermi Energy: {fermi_energy} eV")
    
    energies, occupancies = build_band_arrays(kpts, bands_data)
    E_b = system_type_check(fermi_energy, energies, kpts)

    if E_b == 0:
        print("Band gap = 0.0000 eV")
        print()
        print("System is metallic, no rashba")
    else:
        vbm, cbm, vbm_kpoint, cbm_kpoint, vbm_band, cbm_band = find_vbm_cbm(energies, fermi_energy, kpts)
        print(f"VBM = {vbm:.4f} eV at k-point {kpts[vbm_kpoint]}")
        print(f"CBM = {cbm:.4f} eV at k-point {kpts[cbm_kpoint]}")
        print(f"Band gap = {E_b:.4f} eV")
        print()

        vb_minima, cb_minima = find_extrema(fermi_energy, energies, kpts, k_density)
        if len(cb_minima)==2:
            print(f"Minima near the CBM found at k-point indices: {cb_minima}")
        else:
            print("No other minima found in the specified range near the CBM.")
            
        if len(vb_minima)==2:
            print(f"Maxima near the VBM found at k-point indices: {vb_minima}")
        else:
            print("No other maxima found in the specified range near the VBM.")
        
        find_split(fermi_energy, energies, kpts, hs_points, hs_labels, k_density)
        
    plot_band(fermi_energy, energies, kpts, hs_points, hs_labels,k_density)

if __name__ == "__main__":
    main()
