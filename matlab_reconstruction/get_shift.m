function shift_matrix = get_shift(mode, factor, theta)

    sd = -factor * cos(theta);  
    
    
    if mode
        shift_matrix = [1           0   0            0
                        0           1   0            0
                        sd          0   1            0
                        0           0   0            1];
    else
        shift_matrix = [1           0   0            0
                        0           1   0            0
                        0           sd  1            0
                        0           0   0            1];
    end

end


