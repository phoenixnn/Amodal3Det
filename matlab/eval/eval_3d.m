% performance evaluation (mAP)

dataset_path = '../../dataset/NYUV2';
% detection classes
det_classes = {'background', 'bathtub',  'bed', 'bookshelf', 'box', ... 
               'chair', 'counter', 'desk', 'door', 'dresser', ...
               'garbage bin', 'lamp', 'monitor', 'night stand', ...
               'pillow', 'sink', 'sofa', 'table', 'television', 'toilet'};
 
% load test set
load(fullfile(dataset_path, 'nyusplits.mat'));
imset = tst - 5000;

% load detection results
var = load('../../rgbd_3det/output/all_boxes_cells_test.mat');
all_boxes = var.all_boxes;

% non-maximum suppress
thresh = 0.3;
nms_boxes = apply_nms(all_boxes, thresh);

% classwise detection
gtPrefixPath = '../NYUV2';
APs = zeros(19, 1);
for cls = 1 : 19
    [rec,prec,ap] = d_eval_det_3d(imset, gtPrefixPath, cls, nms_boxes(cls+1, :));
    APs(cls) = ap;

    % plot precision/recall
    f = figure;
    plot(rec, prec,'-');
    grid;
    xlabel 'recall'
    ylabel 'precision'
    title(sprintf('class: %s, dataset: %s, AP = %.3f',det_classes{cls+1}, 'nyu Test',ap));
 
    %% save fig to image
    rez = 120; %resolution (dpi) of final graphic
    figpos = getpixelposition(f); 
    resolution = get(0,'ScreenPixelsPerInch');
    set(f,'paperunits','inches','papersize',figpos(3:4)/resolution,'paperposition',[0 0 figpos(3:4)/resolution]);
    print(f,fullfile('./',[det_classes{cls+1} ,'_ap_3d.png']), ...
          '-dpng',['-r',num2str(rez)],'-opengl')
    close(f);
end

fprintf('the mAP is %f\n', mean(APs));