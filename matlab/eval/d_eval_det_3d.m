
function [rec,prec,ap] = d_eval_det_3d(imset, gtPrefixPath, cls, predicts)
% det thresh
det_thresh = 0.25; 

% load ground truth 
gt(numel(imset)) = struct('BB',[],'det',[]);
npos = 0;
for i = 1 : numel(imset)  
    name = num2str(imset(i));
    % read annotation
    var = load (fullfile(gtPrefixPath, 'gt_3D_19', [name, '.mat']));
    % boxes_3d: [cx, cy, cz, l, w, h, theta (radian)]
    gt_bbox_3d = var.gt_boxes_3d;
    gt_bbox_3d(:, end) = gt_bbox_3d(:, end)*pi/180;
    
    var = load (fullfile(gtPrefixPath, 'gt_label_19',  [name, '.mat']));
    gt_bbox_label = var.gt_class_ids;
    
    % extract objects of class
    clsinds = find(gt_bbox_label == cls);
    gt(i).BB = gt_bbox_3d(clsinds, :);
    gt(i).det = false(numel(clsinds),1);
    
    npos = npos + numel(clsinds);
end

% format detected boxes [ids, score, cx, cy, cz, l, w, h, theta (radian)]
dets = [];
for i = 1 : numel(predicts)
    if ~isempty(predicts{i})
        tmp = predicts{i}(:, 5:12);
        tmp = [ones(size(tmp, 1), 1)*i tmp];
        dets = [dets;  tmp];
    end
end

% sort detections by decreasing confidence
ids = dets(:, 1);
confidence = dets(:, 2);
BB = dets(:, 3:end);
[sc, si] = sort(confidence, 'descend');
ids = ids(si);
BB = BB(si, :);

% assign detections to ground truth objects
nd = numel(confidence);
tp = zeros(nd,1);
fp = zeros(nd,1);

for d = 1 : nd
    
    % find ground truth image
    i = ids(d);
    
    % load Rtilt
    name = num2str(imset(i));
    var = load (fullfile(gtPrefixPath, 'Rtilt',  [name '.mat']));
    Rtilt = var.Rtilt;

    % assign detection to ground truth object if any
    bb = BB(d, :);
    v_bb = bb(4) * bb(5) * bb(6);
    corners_bb = calc_corners_tilt(bb, Rtilt);
    
    ovmax = -inf;
    for j = 1 : size(gt(i).BB, 1)
        bbgt = gt(i).BB(j, :);
        corners_gt = calc_corners_tilt(bbgt, Rtilt);
        v_gt = bbgt(4) * bbgt(5) * bbgt(6);
        iou = calc_IoU_3d(corners_bb, corners_gt, v_bb, v_gt);
        
        if iou > ovmax
            ovmax = iou;
            jmax = j;
        end
    end
    % assign detection as true positive/don't care/false positive
    if ovmax >= det_thresh      
        if ~gt(i).det(jmax)
            tp(d)=1;            % true positive
            gt(i).det(jmax)=true;
        else
            fp(d)=1;            % false positive (multiple detection)
        end
    else
        fp(d)=1;                    % false positive
    end
end

% compute precision/recall
fp = cumsum(fp);
tp = cumsum(tp);
rec = tp/npos;
prec = tp./(fp+tp);

% compute average precision
ap = 0;
for t = 0 : 0.1 : 1
    p=max(prec(rec>=t));
    if isempty(p)
        p=0;
    end
    ap=ap+p/11;
end


end

