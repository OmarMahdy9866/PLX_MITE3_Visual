import numpy as np

def stress_tensor_invariants(principal_stresses):
    """
    Computes the stress tensor invariants (I1, I2, I3)
    given the principal stresses (sigma_1, sigma_2, sigma_3).

    Parameters:
        principal_stresses (tuple or list): Principal stresses (sigma_1, sigma_2, sigma_3)

    Returns:
        dict: Dictionary containing the invariants I1, I2, and I3
    """
    sigma1, sigma2, sigma3 = principal_stresses

    # Compute the invariants
    I1 = sigma1 + sigma2 + sigma3
    I2 = sigma1 * sigma2 + sigma2 * sigma3 + sigma3 * sigma1
    I3 = sigma1 * sigma2 * sigma3

    return {"I1": I1, "I2": I2, "I3": I3}


def deviatoric_stress_tensor_invariants(principal_stresses):
    """
    Computes the deviatoric stress tensor invariants (J2, J3)
    given the principal stresses (sigma_1, sigma_2, sigma_3).

    Parameters:
        principal_stresses (tuple or list): Principal stresses (sigma_1, sigma_2, sigma_3)

    Returns:
        dict: Dictionary containing the invariants J2 and J3
    """
    sigma1, sigma2, sigma3 = principal_stresses

    # Compute the mean stress (hydrostatic stress)
    mean_stress = (sigma1 + sigma2 + sigma3) / 3.0

    # Compute the deviatoric stresses
    s1 = sigma1 - mean_stress
    s2 = sigma2 - mean_stress
    s3 = sigma3 - mean_stress

    # Compute the invariants
    J2 = (1 / 6) * ((s1 - s2) ** 2 + (s2 - s3) ** 2 + (s3 - s1) ** 2)
    J3 = s1 * s2 * s3

    return {"J2": J2, "J3": J3}

# Principal stresses
principal_stresses = (100, 50, 25)

# Compute stress tensor invariants
stress_invariants = stress_tensor_invariants(principal_stresses)
print("Stress Tensor Invariants:", stress_invariants)

# Compute deviatoric stress tensor invariants
deviatoric_invariants = deviatoric_stress_tensor_invariants(principal_stresses)
print("Deviatoric Stress Tensor Invariants:", deviatoric_invariants)

