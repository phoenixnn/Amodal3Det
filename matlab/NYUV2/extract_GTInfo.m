
% extract groundtruth information

anno_path = '../../dataset/NYUV2/annotations';
dataset_path = '../../dataset/NYUV2';

% detection classes
det_classes = {'bathtub',  'bed', 'bookshelf', 'box', 'chair', 'counter', ... 
               'desk', 'door', 'dresser', 'garbage bin', 'lamp', ...
               'monitor', 'night stand', 'pillow', 'sink', 'sofa', ...
               'table', 'television', 'toilet'};


for i = 1 : 1449
   fprintf('%d \n', i);
   
   % load annotations
   load(fullfile(anno_path, [num2str(i) '.mat']));
   [mask_valid_class, valid_idx] = ismember(data.gtLabelNames, det_classes);
   mask_labeled = ~cellfun(@isempty, data.gt3D);
   mask = mask_valid_class & mask_labeled;
   gt3D = data.gt3D(mask);
   gt_label_names = data.gtLabelNames(mask);
   % 0 - background class
   gt_class_ids = valid_idx(mask);
   save(fullfile('gt_label_19', [num2str(i) '.mat']), 'gt_class_ids');
   
   % gt boxes 2d for p/n selection
   gt2Dsel = data.gt2Dsel(mask);
   gt_boxes_sel = [];
   for j = 1 : numel(gt2Dsel)
      gt_boxes_sel = cat(1, gt_boxes_sel, gt2Dsel{j}); 
   end
   gt_boxes_sel = gt_boxes_sel - 1;
   save(fullfile('gt_2Dsel_19', [num2str(i) '.mat']), 'gt_boxes_sel');
   
   % gt boxes 3d
   gt_boxes_3d = VectorizeBboxes(gt3D, data.Rtilt);
   save(fullfile('gt_3D_19', [num2str(i) '.mat']), 'gt_boxes_3d');
   
   % Rtilt
   Rtilt = data.Rtilt;
   save(fullfile('Rtilt', [num2str(i) '.mat']), 'Rtilt');

end