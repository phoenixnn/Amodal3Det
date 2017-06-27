
num_cls = 20; % include background class

for i = 1 : 1449
   fprintf('%d \n', i);
   
   % load gt_2Dsel_19
   load(fullfile('gt_2Dsel_19', [num2str(i), '.mat']));
   
   if isempty(gt_boxes_sel)
       continue;
   end
   
   % load proposal 2d
   load(fullfile('proposal2d', [num2str(i), '.mat']));  
   % combine 
   boxes = cat(1, gt_boxes_sel, boxes2d_prop);
   % load gt_label_19
   load(fullfile('gt_label_19', [num2str(i), '.mat']));  
   %
   gt_overlaps = compute_box2d_overlaps(boxes, gt_boxes_sel, gt_class_ids, num_cls);
   
   save(fullfile('gt_overlaps_19', [num2str(i), '.mat']), 'gt_overlaps');
   
end