

function gt_overlaps = compute_box2d_overlaps(boxes, gt_boxes, gt_classes, num_cls)
   
   num_box = size(boxes, 1);
   num_gt_box = size(gt_boxes, 1);
   overlaps = zeros(num_box, num_gt_box);
   
   for i = 1 : num_box
        for j = 1 : num_gt_box
           overlaps(i, j) = overlap_ratio(boxes(i,:), gt_boxes(j,:)); 
        end
   end
   
   [max_overlaps, ids] = max(overlaps, [], 2);
   gt_overlaps = zeros(num_box, num_cls);
   
   % index start at 1
   labels = gt_classes(ids) + 1;  
   if size(labels, 1) > size(labels, 2)
       labels = labels';
   end
   inds = sub2ind(size(gt_overlaps), 1:num_box, labels);
   
   gt_overlaps(inds) = max_overlaps;

end

function IoU = overlap_ratio(box, gt_box)
    
    x_min = max(box(1), gt_box(1));
    y_min = max(box(2), gt_box(2));
    x_max = min(box(3), gt_box(3));
    y_max = min(box(4), gt_box(4));
    
    if (x_min > x_max) || (y_min > y_max)
        IoU = 0;
        return;
    end
    
    A1 = (box(3) - box(1) + 1) * (box(4) - box(2) + 1);
    A2 = (gt_box(3) - gt_box(1) + 1) * (gt_box(4) - gt_box(2) + 1);
    common = (x_max - x_min + 1) * (y_max -y_min + 1);
    IoU = common/(A1 + A2 - common);

end