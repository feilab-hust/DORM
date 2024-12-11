function affine_matrix = get_affine(mode, factor, theta, iso)
    % 用于移位加旋转

    if iso
        a = 1;
    else
        a = 1/sin(theta)/factor;
    end
    if mode
        affine_matrix = [-cos(theta)*a   0  sin(theta)   0
                         0               1  0            0
                         factor*a        0  0            0
                         0               0  0            1];
    else
        affine_matrix = [0               1  0            0
                         -cos(theta)*a   0  sin(theta)   0
                         factor*a        0  0            0
                         0               0  0            1];
    end

end

