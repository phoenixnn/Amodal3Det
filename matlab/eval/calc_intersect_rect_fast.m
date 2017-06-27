function A = calc_intersect_rect_fast( rect1, rect2 )

lineseg1 = [rect1(1,:) rect1(2,:);
            rect1(2,:) rect1(3,:);
            rect1(3,:) rect1(4,:);
            rect1(4,:) rect1(1,:)];
        
lineseg2 = [rect2(1,:) rect2(2,:);
            rect2(2,:) rect2(3,:);
            rect2(3,:) rect2(4,:);
            rect2(4,:) rect2(1,:)];
 % interctions       
 out = lineSegmentIntersect(lineseg1, lineseg2);
 x = out.intMatrixX(out.intAdjacencyMatrix);
 y = out.intMatrixY(out.intAdjacencyMatrix);
 intersects = unique([x y], 'rows');
 
 % included vertices
 isIn_12 = d_inRect(rect1, rect2);
 isIn_21 = d_inRect(rect2, rect1);
 
 v1 = rect1(isIn_12, :);
 v2 = rect2(isIn_21, :); 
 pts = [intersects; v1; v2];

 % re-sort vertices
 cx = mean(pts(:,1));
 cy = mean(pts(:,2));
 a = atan2(pts(:,2) - cy, pts(:,1) - cx);
 [~, order] = sort(a);
 pts = pts(order, :);
 
 A = polyarea(pts(:,1), pts(:,2));

end

