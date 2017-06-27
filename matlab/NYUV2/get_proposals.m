% compute proposals for both 2D and 3D
%

dataset_path = '../../dataset/NYUV2';

% detection classes
det_classes = {'bathtub',  'bed', 'bookshelf', 'box', 'chair', 'counter', ...
               'desk', 'door', 'dresser', 'garbage bin', 'lamp', ...
               'monitor', 'night stand', 'pillow', 'sink', 'sofa', ... 
               'table', 'television', 'toilet'};
 
% average box dimensions
load(fullfile(dataset_path,'classwise_bboxDims.mat'));

% intrinsic matrix
K = GetIntrinsicMat('nyu_color_standard_crop');

for i = 1 : 1449
    imPackName = sprintf('NYU%.4d', i);
    fprintf('%s \n', imPackName);
    
    % 2d proposals (format: [xmin, ymin, xmax, ymax], start at 0)
    var = load(fullfile('Segs', imPackName, 'candidates.mat'));
    candidates = var.candidates;
    bbox_2d = candidates.bboxes;
    boxes = bbox_2d(:, [2,1,4,3]) - 1;
    % remove rois whos area is less than 200 pixels
    w = boxes(:, 3) - boxes(:, 1) + 1;
    h = boxes(:, 4) - boxes(:, 2) + 1;
    A = w.*h;
    valid = find(A >= 200);
    boxes2d_prop = boxes(valid, :);
    save(fullfile('proposal2d', [num2str(i), '.mat']), 'boxes2d_prop');
    
    % 3d proposals
    sp_base = candidates.superpixels;
    sp_label = candidates.labels;
    sp_label = sp_label(valid);
    
    % load hole-filled depth map
    load(fullfile(dataset_path, 'dmap_f', [num2str(i), '.mat']));
    depth_fill = double(dmap_f);
    % load original map
    load(fullfile(dataset_path, 'rawDepth', [num2str(i), '.mat']));
    % 
    boxes3d_prop = get_3d_proposals(rawDepth, K, sp_base, sp_label, depth_fill, avg_boxDims_cls);
    save(fullfile('proposal3d', [num2str(i), '.mat']), 'boxes3d_prop');
    
end