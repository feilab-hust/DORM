function [img_l1,img_l2, img_l3] = crop_lens(img, y_center, bg_mean, bg_std, rotate_data, h_l2,w_l2,x_dis1,x_dis2,x_middle_l2)

    w_camera = size(img, 2);
    d = size(img, 3);
    img_l1 = uint16(normrnd(bg_mean, bg_std, [h_l2, w_l2,d]));
    img_l2 = uint16(normrnd(bg_mean, bg_std,  [h_l2, w_l2,d]));
    img_l3 = uint16(normrnd(bg_mean, bg_std,  [h_l2, w_l2,d]));
    

    y_l2 = y_center(2) - floor(h_l2 / 2);
    x_l2 = x_middle_l2 - floor(w_l2 / 2);
    img_l2_r = img(:, x_l2:x_l2 + w_l2 - 1, :);
    if rotate_data
        img_l2_r = imrotate(img_l2_r, 0.23, 'bilinear', 'crop');
    end
    img_l2(:, :, :) = img_l2_r(y_l2:y_l2 + h_l2 - 1, :, :);
    
 
    y_l1 = max(1, y_center(1) - floor(h_l2 / 2));
    x_l1 = max(1, x_middle_l2 - x_dis1 - floor(w_l2 / 2));
    img_l1_r = img(:, x_l1:x_l1 + w_l2 - 1, :);
    if rotate_data
        img_l1_r = imrotate(img_l1_r, 0.7, 'bilinear', 'crop');
    end
    img_l1(:, :, :) = img_l1_r(y_l1:y_l1 + h_l2 - 1, :, :);
    

    y_l3 = max(1, y_center(3) - floor(h_l2 / 2));
    x_l3 = min(w_camera, x_middle_l2 + x_dis2 - floor(w_l2 / 2));
    img_l3_r = img(:, x_l3:x_l3 + w_l2 - 1, :);
%     if rotate_data
%         img_l3_r = imrotate(img_l3_r, 0.17, 'bilinear', 'crop');
%     end
    img_l3(:, :, :) = img_l3_r(y_l3:y_l3 + h_l2 - 1, :, :);
end