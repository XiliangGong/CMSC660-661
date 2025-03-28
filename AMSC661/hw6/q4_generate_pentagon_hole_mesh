function [vert, tria] = generate_pentagon_hole_mesh()
%GENERATE_PENTAGON_HOLE_MESH - mesh a 5-gon with a smaller hole inside

    phi1 = (0:4)' / 5 * 2 * pi + pi / 10;
    pfix1 = 1.5 * [cos(phi1), sin(phi1)];

    phi2 = (0:4)' / 5 * 2 * pi - pi / 10;
    r = sin(pi / 5) / sind(108 / 2) * cosd(72) / cosd(108 / 2);
    pfix2 = 1.5 * r * [cos(phi2), sin(phi2)];

    node = [pfix1; pfix2];

    edge1 = [(1:5)', [2:5 1]'];            
    edge2 = [(6:10)', [7:10 6]'];        
    edge = [edge1; edge2];


    hfun = 0.12;

    [vert, ~, tria, ~] = refine2(node, edge, [], [], hfun);

    figure;
    patch('faces',tria(:,1:3), 'vertices',vert, ...
        'facecolor','w', ...
        'edgecolor',[.2, .2, .2]);
    hold on; axis image off;
    patch('faces',edge(:,1:2), 'vertices',node, ...
        'edgecolor','k', 'linewidth', 1.5);
    title('Pentagon with Inner Hole Mesh');
end
