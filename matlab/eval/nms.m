function keep = nms( dets, thresh )
    
    x1 = dets(:, 1);
    y1 = dets(:, 2);
    x2 = dets(:, 3);
    y2 = dets(:, 4);
    scores = dets(:, 5);

    areas = (x2 - x1 + 1) .* (y2 - y1 + 1);
    
    % index in descending order w.r.t scores
    [~, order] = sort(scores, 'descend');
    keep = []; 
    
    while numel(order) > 0
        % find the first index
        i = order(1);
        
        % update keep
        keep = [keep; i];
        
        % find overlap bound
        xx1 = max(x1(i), x1(order(2:end)));
        yy1 = max(y1(i), y1(order(2:end)));
        xx2 = min(x2(i), x2(order(2:end)));
        yy2 = min(y2(i), y2(order(2:end)));
        
        % dims and IoU
        w = max(0.0, xx2 - xx1 + 1);
        h = max(0.0, yy2 - yy1 + 1);
        inter = w .* h;
        ovr = inter ./ (areas(i) + areas(order(2:end)) - inter); 

        inds = find(ovr <= thresh);
        order = order(inds + 1);
    end
end

