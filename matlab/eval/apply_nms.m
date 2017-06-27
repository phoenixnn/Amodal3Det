% all_boxes: cells n_cls x n_img

function nms_boxes = apply_nms( all_boxes, thresh )
    [num_classes, num_images] = size(all_boxes);
    nms_boxes = cell(num_classes, num_images);
    
    for cls_ind = 1 : num_classes
        for im_ind = 1 : num_images
            dets = all_boxes{cls_ind, im_ind};
            if isempty(dets)
                continue;
            end
            
            % apply nms
            keep = nms(dets, thresh);
            if isempty(keep)
                continue;
            end
            
            nms_boxes{cls_ind, im_ind} = dets(keep, :);
        end
    end
end

