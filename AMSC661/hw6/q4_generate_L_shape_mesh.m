function [vert, tria] = generate_L_shape_mesh()
%GENERATE_L_SHAPE_MESH Create mesh for an L-shaped domain using refine2.
% Output:
%   vert - coordinates of mesh vertices
%   tria - triangle connectivity list

    % Define outer L-shaped polygon (counter-clockwise)
    node = [
        -1.5, -1.5;
         1.5, -1.5;
         1.5,  0.0;
         0.0,  0.0;
         0.0,  1.5;
        -1.5,  1.5
    ];

    edge = [
        1, 2;
        2, 3;
        3, 4;
        4, 5;
        5, 6;
        6, 1
    ];

    hfun = 0.1;  % target edge length

    [vert, ~, tria, ~] = refine2(node, edge, [], [], hfun);

    % Optional: visualize
    figure;
    patch('faces',tria(:,1:3), 'vertices',vert, ...
        'facecolor','w', ...
        'edgecolor',[.2, .2, .2]);
    hold on; axis image off;
    patch('faces',edge(:,1:2), 'vertices',node, ...
        'edgecolor','k', 'linewidth', 1.5);
    title(['L-shape Mesh']);
end




