import torch
from sklearn.decomposition import PCA

def lavg(in_scat_coeffs, J=3, L=8, m=2):
    img_h, img_w = in_scat_coeffs.shape[-2], in_scat_coeffs.shape[-1]
    # Initialize the mean and standard deviation output tensors
    if m == 2:
        output_sections = J * (J - 1) // 2 + J + 1
    else:
        output_sections = J + 1
    avg_tensor = torch.zeros((output_sections, img_h, img_w))
    std_tensor = torch.zeros_like(avg_tensor)
    # Process Zeroth order directly across all images
    if in_scat_coeffs.dim() == 4:
        avg_tensor[0] = in_scat_coeffs[:, 0].mean(dim=0)
        std_tensor[0] = in_scat_coeffs[:, 0].std(dim=0)
    else:
        avg_tensor[0] = in_scat_coeffs[0]
        std_tensor[0] = torch.zeros_like(avg_tensor[0])
    # First order averaging
    for j in range(1, J + 1):
        start_idx = (j - 1) * L + 1
        end_idx = j * L + 1
        section = in_scat_coeffs[:, start_idx:end_idx] if in_scat_coeffs.dim() == 4 else in_scat_coeffs[start_idx:end_idx]
        avg_tensor[j] = section.mean(dim=(0, -3)) if in_scat_coeffs.dim() == 4 else section.mean(dim=-3)
        std_tensor[j] = section.std(dim=(0, -3)) if in_scat_coeffs.dim() == 4 else section.std(dim=-3)
    if m == 2:
        num_coeffs = in_scat_coeffs.shape[1] if in_scat_coeffs.dim() == 4 else in_scat_coeffs.shape[0]
        for i in range(J * (J - 1) // 2):
            for l in range(L):
                start_idx = i * L + l * L + J * L + 1
                end_idx = start_idx + L
                if end_idx > num_coeffs:
                    continue  # Skip this loop if the end index exceeds the number of coefficients
                section = in_scat_coeffs[:, start_idx:end_idx] if in_scat_coeffs.dim() == 4 else in_scat_coeffs[start_idx:end_idx]
                sum_tensor = section.sum(dim=(0, -3)) if in_scat_coeffs.dim() == 4 else section.sum(dim=-3)
                variance = section.var(dim=(0, -3), unbiased=False) if in_scat_coeffs.dim() == 4 else section.var(dim=-3, unbiased=False)
                if torch.isnan(variance).any(): raise Exception("Variance is NaN!")
                sum_sq_diff = variance * (end_idx - start_idx)
                avg_tensor[i + J + 1] = sum_tensor / float(L)
                std_tensor[i + J + 1] = torch.sqrt(sum_sq_diff / float(L))
    return avg_tensor.squeeze(0), std_tensor.squeeze(0)


def adj_lavg(in_scat_coeffs, J=3, L=8, m=2):
    if L % 2 != 0:
        raise ValueError("L must be even for proper l-averaging.")

    # Determine the correct dimension indexes based on tensor shape
    batch_mode = in_scat_coeffs.dim() == 4  # True if batch of images
    num_imgs = in_scat_coeffs.shape[0] if batch_mode else 1
    img_h, img_w = in_scat_coeffs.shape[-2], in_scat_coeffs.shape[-1]
    
    # Calculate output sections for first and second order terms
    first_order_sections = J * (L // 2) + 1
    second_order_sections = J * (J - 1) // 2 * (L // 2) ** 2 if m == 2 else 0
    output_sections = first_order_sections + second_order_sections
    avg_tensor = torch.zeros((num_imgs, output_sections, img_h, img_w))
    std_tensor = torch.zeros_like(avg_tensor)
    
    # Process Zeroth order directly across all images
    avg_tensor[:, 0] = in_scat_coeffs[:, 0].mean(dim=0) if batch_mode else in_scat_coeffs[0]
    std_tensor[:, 0] = in_scat_coeffs[:, 0].std(dim=0) if batch_mode else torch.zeros_like(avg_tensor[:, 0])

    # Adjacent l averaging for first order
    idx = 1
    for j in range(1, J + 1):
        for l in range(0, L, 2):  # Step by 2 for pairs l and l+1
            start_idx1 = (j - 1) * L + l + 1
            start_idx2 = start_idx1 + 1
            indices = slice(start_idx1, start_idx2 + 1)
            section = in_scat_coeffs[:, indices] if batch_mode else in_scat_coeffs[indices]
            avg_tensor[:, idx] = section.mean(dim=(0, -3) if batch_mode else -3)
            std_tensor[:, idx] = section.std(dim=(0, -3) if batch_mode else -3)
            idx += 1

    # Second order averaging if m == 2
    if m == 2:
        idx = first_order_sections  # Start indexing after the first order sections
        for j1 in range(J):
            for l1 in range(0, L, 2):  # Iterate over pairs l1 and l1+1
                for j2 in range(j1 + 1, J):
                    for l2 in range(0, L, 2):  # Iterate over pairs l2 and l2+1
                        start_idx = ((j1 * J + j2) * (L // 2) ** 2 + (l1 // 2) * (L // 2) + (l2 // 2)) * 2 + 1
                        end_idx = start_idx + 2
                        if end_idx > in_scat_coeffs.shape[1]:
                            continue  # Skip if the end index exceeds the number of coefficients
                        section = in_scat_coeffs[:, start_idx:end_idx] if batch_mode else in_scat_coeffs[start_idx:end_idx]
                        sum_tensor = section.sum(dim=(0, 1) if batch_mode else 0)
                        variance = section.var(dim=(0, 1), unbiased=False) if batch_mode else section.var(dim=0, unbiased=False)
                        if torch.isnan(variance).any(): raise Exception("Variance is NaN!")
                        sum_sq_diff = variance * (end_idx - start_idx)
                        avg_tensor[:, idx] = sum_tensor / float(2)  # Dividing by 2 for adjacent pairs
                        std_tensor[:, idx] = torch.sqrt(sum_sq_diff / float(2))
                        idx += 1

        # Return the final averaged tensors
        return avg_tensor.squeeze(0), std_tensor.squeeze(0)
    

def ldiff(in_scat_coeffs, J=3, L=8, m=2):
    """
    Reduce scattering coefficients by taking the difference between adjacent l pairs.
    """
    
    if L < 2:
        raise ValueError("L must be at least 2 for adjacent l differences.")

    # Determine the correct dimension indexes based on tensor shape
    batch_mode = in_scat_coeffs.dim() == 4  # True if batch of images
    num_imgs = in_scat_coeffs.shape[0] if batch_mode else 1
    img_h, img_w = in_scat_coeffs.shape[-2], in_scat_coeffs.shape[-1]
    
    # Calculate output sections for first and second order terms
    first_order_sections = J * (L - 1) + 1
    second_order_sections = J * (J - 1) // 2 * (L - 1) ** 2 if m == 2 else 0
    output_sections = first_order_sections + second_order_sections
    avg_tensor = torch.zeros((num_imgs, output_sections, img_h, img_w))
    std_tensor = torch.zeros_like(avg_tensor)
    
    # Process Zeroth order directly across all images
    avg_tensor[:, 0] = in_scat_coeffs[:, 0].mean(dim=0) if batch_mode else in_scat_coeffs[0]
    std_tensor[:, 0] = in_scat_coeffs[:, 0].std(dim=0) if batch_mode else torch.zeros_like(avg_tensor[:, 0])

    # Adjacent l difference for first order
    idx = 1
    for j in range(1, J + 1):
        for l in range(L - 1):  # For adjacent l pairs
            start_idx1 = (j - 1) * L + l + 1
            start_idx2 = start_idx1 + 1
            section1 = in_scat_coeffs[:, start_idx1] if batch_mode else in_scat_coeffs[start_idx1]
            section2 = in_scat_coeffs[:, start_idx2] if batch_mode else in_scat_coeffs[start_idx2]
            diff_section = section1 - section2
            
            avg_tensor[:, idx] = diff_section.mean(dim=0) if batch_mode else diff_section
            std_tensor[:, idx] = diff_section.std(dim=0) if batch_mode else torch.zeros_like(avg_tensor[:, idx])
            idx += 1

    # Second order adjacent l differences if m == 2
    if m == 2:
        idx = first_order_sections  # Start indexing after the first order sections
        for j1 in range(J):
            for l1 in range(L - 1):  # Iterate over adjacent l1 pairs
                for j2 in range(j1 + 1, J):
                    for l2 in range(L - 1):  # Iterate over adjacent l2 pairs
                        start_idx1 = ((j1 * J + j2) * (L - 1) ** 2 + l1 * (L - 1) + l2) * 2 + 1
                        end_idx1 = start_idx1 + 1
                        start_idx2 = end_idx1
                        end_idx2 = start_idx2 + 1
                        if end_idx2 > in_scat_coeffs.shape[1]:
                            continue  # Skip if the end index exceeds the number of coefficients
                        section1 = in_scat_coeffs[:, start_idx1] if batch_mode else in_scat_coeffs[start_idx1]
                        section2 = in_scat_coeffs[:, end_idx2] if batch_mode else in_scat_coeffs[end_idx2]
                        diff_section = section1 - section2
                        
                        avg_tensor[:, idx] = diff_section.mean(dim=0) if batch_mode else diff_section
                        std_tensor[:, idx] = diff_section.std(dim=0) if batch_mode else torch.zeros_like(avg_tensor[:, idx])
                        idx += 1

    # Return the final tensors with the differences
    return avg_tensor.squeeze(0), std_tensor.squeeze(0)

def spatial_red_lavg(in_scat_coeffs, J=3, L=8, m=2):
                
                # Initialize the mean and standard deviation output tensors
                if m == 2:
                    output_sections = J * (J - 1) // 2 + J + 1
                else:
                    output_sections = J + 1

                # Initialize the avg_tensor and std_tensor with the appropriate dimensions
                avg_tensor = torch.zeros(output_sections, in_scat_coeffs.size(-1))  # Now it's 2D
                std_tensor = torch.zeros_like(avg_tensor)

                # Process Zeroth order directly across all images
                in_scat_coeffs = torch.mean(in_scat_coeffs, dim=(-2, -1))
                if in_scat_coeffs.dim() == 2:  # [batch, num_coeffs]
                    avg_tensor[0] = in_scat_coeffs[0].mean(dim=0)
                    std_tensor[0] = in_scat_coeffs[0].std(dim=0)
                else:
                    avg_tensor[0] = in_scat_coeffs[0]
                    std_tensor[0] = torch.zeros_like(avg_tensor[0])  # Initialize std_tensor[0] as zeros


                # First order averaging
                for j in range(1, J + 1):
                    start_idx = (j - 1) * L + 1
                    end_idx = j * L + 1
                    section = in_scat_coeffs[:, start_idx:end_idx] if in_scat_coeffs.dim() == 2 else in_scat_coeffs[start_idx:end_idx]
                    avg_tensor[j] = section.mean(dim=(0, 1)) if in_scat_coeffs.dim() == 2 else section.mean(dim=-1)
                    std_tensor[j] = section.std(dim=(0, 1)) if in_scat_coeffs.dim() == 2 else section.std(dim=-1)

                if m == 2:
                    num_coeffs = in_scat_coeffs.shape[-1]
                    for i in range(J * (J - 1) // 2):
                        for l in range(L):
                            start_idx = i * L + l * L + J * L + 1
                            end_idx = start_idx + L
                            if end_idx > num_coeffs:
                                continue  # Skip this loop if the end index exceeds the number of coefficients

                            section = in_scat_coeffs[:, start_idx:end_idx] if in_scat_coeffs.dim() == 2 else in_scat_coeffs[start_idx:end_idx]
                            sum_tensor = section.sum(dim=(0, -1)) if in_scat_coeffs.dim() == 2 else section.sum(dim=-1)
                            variance = section.var(dim=(0, -1), unbiased=False) if in_scat_coeffs.dim() == 2 else section.var(dim=-1, unbiased=False)
                            if torch.isnan(variance).any(): raise Exception("Variance is NaN!")

                            sum_sq_diff = variance * (end_idx - start_idx)
                            avg_tensor[i + J + 1] = sum_tensor / float(L)
                            std_tensor[i + J + 1] = torch.sqrt(sum_sq_diff / float(L))

                return avg_tensor.squeeze(0), std_tensor.squeeze(0)
            
    
def javg(in_scat_coeffs, J=3, L=8, m=2, scale_groups=2):
    """
    Reduce scattering coefficients by aggregating across scales.
    
    Parameters:
    - in_scat_coeffs: The input scattering coefficients.
    - J: Number of scales.
    - L: Number of orientations.
    - m: Order of scattering (1 or 2).
    - scale_groups: Number of scales to aggregate into one group.
    
    Returns:
    - avg_tensor: Aggregated mean tensor.
    - std_tensor: Aggregated standard deviation tensor.
    """
    img_h, img_w = in_scat_coeffs.shape[-2], in_scat_coeffs.shape[-1]
    
    # Determine the number of aggregated scale groups
    num_scale_groups = (J + scale_groups - 1) // scale_groups
    
    # Initialize output tensors
    avg_tensor = torch.zeros((num_scale_groups, img_h, img_w))
    std_tensor = torch.zeros_like(avg_tensor)
    
    # Process Zeroth order directly
    if in_scat_coeffs.dim() == 4:
        avg_tensor[0] = in_scat_coeffs[:, 0].mean(dim=0)
        std_tensor[0] = in_scat_coeffs[:, 0].std(dim=0)
    else:
        avg_tensor[0] = in_scat_coeffs[0]
        std_tensor[0] = torch.zeros_like(avg_tensor[0])
    
    # Aggregating across scales
    for group in range(1, num_scale_groups):
        start_idx = (group - 1) * scale_groups * L + 1
        end_idx = min(group * scale_groups * L + 1, in_scat_coeffs.shape[1])
        
        section = in_scat_coeffs[:, start_idx:end_idx] if in_scat_coeffs.dim() == 4 else in_scat_coeffs[start_idx:end_idx]
        
        # Aggregate the section by taking the mean and standard deviation
        avg_tensor[group] = section.mean(dim=(0, -3)) if in_scat_coeffs.dim() == 4 else section.mean(dim=-3)
        std_tensor[group] = section.std(dim=(0, -3)) if in_scat_coeffs.dim() == 4 else section.std(dim=-3)
    
    return avg_tensor.squeeze(0), std_tensor.squeeze(0)


def pca(in_scat_coeffs, J=3, L=8, m=2, n_components=5):
    flat_coeffs = in_scat_coeffs.view(in_scat_coeffs.size(0), -1)

    # Mean center the data
    flat_coeffs_mean = flat_coeffs.mean(dim=0, keepdim=True)
    flat_coeffs_centered = flat_coeffs - flat_coeffs_mean

    # Apply PCA to the centered data
    U, S, V = torch.svd(flat_coeffs_centered)
    reduced_coeffs = torch.matmul(flat_coeffs_centered, V[:, :n_components])
    num_elements = reduced_coeffs.numel()
    new_shape = (in_scat_coeffs.size(0), n_components, num_elements // (in_scat_coeffs.size(0) * n_components))

    if num_elements != new_shape[0] * new_shape[1] * new_shape[2]:
        raise ValueError("The number of elements does not match, unable to reshape.")

    reduced_tensor = reduced_coeffs.view(*new_shape)

    # Compute statistics on the reduced tensor
    mean_tensor = reduced_tensor.mean(dim=0)
    std_tensor = reduced_tensor.std(dim=0)

    return mean_tensor, std_tensor