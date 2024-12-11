function stack = read_stack(data_path, data_name, stack_size_list, name_list, mode, height, width, bitdepth)
n_slice  = size(name_list, 2);  

name_code = cell(1,n_slice);   
slice_code = (1:n_slice)*0;
img_num = size(stack_size_list);

for i = 1:n_slice
    for j = 1:img_num
        if name_list(i)<=sum(stack_size_list(1:j))
            name_code{i} = data_name{j};
            slice_code(i) = name_list(i)-sum(stack_size_list(1:j-1));
            break
        end
    end
end

if mode   
    stack = zeros(height, width, n_slice, 1);
elseif bitdepth == 48 
    stack = zeros(height, width, n_slice, 2);
else   
    stack = zeros(height, width, n_slice, 1);
end

for i = 1:n_slice
    if mode
        stack(:,:,i,:) = imread(fullfile(data_path, name_code{i}), slice_code(i));
    elseif bitdepth == 48
        img_up = imread(fullfile(data_path, name_code{i}), 1);
        img_down = imread(fullfile(data_path, name_code{i}), 2);
        stack(:,:,i,:) = cat(4, img_up, img_down);
    else
        stack(:,:,i,:) = imread(fullfile(data_path, name_code{i}), slice_code(i));
    end
end

