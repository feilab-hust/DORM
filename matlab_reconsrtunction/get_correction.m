function correction_matrix = get_correction(mode, factor, theta)
    % 用于旋转，旋转矩阵

    c = cos(theta);
    s = sin(theta);


    
    if mode
        correction_matrix = [-c              0  s            0
                             0               1  0            0
                             s               0  c            0
                             0               0  0            1];
    else
        correction_matrix = [0               1  0            0
                             -c              0  s            0
                             s               0  c            0
                             0               0  0            1];
    end

end
