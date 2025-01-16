% Load the edge list into the MATLAB environment
load("graph_edges.txt");
% Use the edge list as a parameter so as to start performing tasks on it
a1_20349877(X21rk74)

function set12 = a1_20349877(elist)
%
% IN:
%     elist - Mx2 array of edges, each row is a pair of vertices
% OUT:
%     set12 - Nx1 vertex clsutering, -1 for SET1 and +1 for SET2

    % Problem size: number of vertices in the graph
    n = max(elist(:));

    % Initialize a square matrix with its elements set to 0
    A = zeros(n);
    % Iterate through all rows within the elist array
    for i = 1:size(elist, 1)
    % Extract the vertices for each edge
        vertex1 = elist(i, 1);
        vertex2 = elist(i, 2);
    
    % Update the adjacency matrix
        A(vertex1, vertex2) = 1;
        A(vertex2, vertex1) = 1;
    end

    % Calculate the degree matrix by creating a diagonal matrix using the
    % adjacency matrix
    degree_matrix = diag(sum(A,2));
    % Create the laplacian matrix
    laplacian_matrix = degree_matrix - A;
    % Compute the eigenvectors and eigenvalues of the Laplacian matrix
    [eigenvectors, eigenvalues] = eig(laplacian_matrix);
    % Extract the second smallest eigenvector or the Fiedler vector
    fiedler_vector = eigenvectors(:,2);
    % Create a binary vector set12 based on whether the values of the
    % Fiedler vector are greater than or equal to 0
    set12 = fiedler_vector >= 0;
    % Convert the binary vector to a clustering vector with values of 1 and
    % -1
    set12 = -((set12 * 2) - 1);

    % Assuming your vector is named 'ans'
    % Create an empty set in order to cluster the binary values of -1
    set1_vertices = [];
    % Iterate through the elements of set12
    for idx = 1:length(set12)
        % Check is an element's value within set12 is -1
        if set12(idx) == -1
            % If so, append the element into the empty set created in order
            % to store these values
            set1_vertices = [set1_vertices, idx];
        end
    end

    % Create an empty set in order to cluster the binary values of 1
    set2_vertices = [];
    % Iterate through the elements of set12
    for jdx = 1:length(set12)
        % Check is an element's value within set12 is 1
        if set12(jdx) == 1
             % If so, append the element into the empty set created in order
            % to store these values
            set2_vertices = [set2_vertices, jdx];
        end
    end

    % Print the vertices in set1
    disp('Set 1 vertices are:');
    disp([set1_vertices]);
    % Print the vertices in set2
    disp('Set 2 vertices are:');
    disp([set2_vertices]);


    % Plot the graph, Cartesian and clustered
    plot271a1(A, set12);
end

function plot271a1(Amat, cvec)
% PLOTCLUSTER(AMAT,CVEC) plots the adjacency matrix AMAT twice;
% first, as a Cartesian grid, and seconnd, by using binary clusters
% in CVEC to plot the graph of AMAT based on two circles
%
% INPUTS: 
%         Amat - NxN adjacency matrix, symmetric with binary entries
%         cvec - Nx1 vector of class labels, having 2 distinct values
% OUTPUTS:
%         none
% SIDE EFFECTS:
%         Plots into the current figure

    % %
    % % Part 1 of 2: plot the graph as a rectangle
    % %

    % Problem size
    [m n] = size(Amat);

    % Factor the size into primes and use the largest as the X size
    nfact = factor(n);
    nx = nfact(end);
    ny = round(n/nx);

    % Create a grid and pull apart into coordinates; offset Y by +2
    [gx, gy] = meshgrid((1:nx) - round(nx/2), (1:ny) + 2);

    % Offset the odd rows to diagram the connections a little better
    for ix=1:2:ny
        gx(ix, :) = gx(ix, :) + 0.25*ix;
    end

    % The plot function needs simple vectors to create the graph
    x = gx(:);
    y = flipud(gy(:));

    % Plot the graph of A using the Cartesian grid
    plot(graph(tril(Amat, -1), 'lower'), 'XData', x, 'YData', y);
    axis('equal');

    % %
    % % Part 2 of 2: plot the graph as pair of circles
    % %
    % Set up the X and Y coordinates of each graph vertex
    xy = zeros(2, numel(cvec));

    % Number of cluster to process
    kset = unique(cvec);
    nk = numel(kset);

    % Base circle is radius 2, points are centers of clusters
    bxy = 2*circlen(nk);

    % Process each cluster
    for ix = 1:nk
        jx = cvec==kset(ix);
        ni = sum(jx);
        xy(:, jx) = bxy(:, ix) + circlen(ni);
    end

    hold on;
    plot(graph(Amat), 'XData', xy(1,:), 'YData', xy(2,:));
    hold off;
    title(sprintf('Clusters of (%d,%d) nodes', ...
        sum(cvec==kset(1)), sum(cvec==kset(2))));
end

function xy = circlen(n)
% XY=CIRCLEN(N) finds N 2D points on a unit circle
%
% INPUTS:
%         N  - positive integer, number of points
% OUTPUTS:
%         XY - 2xN array, each column is a 2D point

    xy = [cos(2*pi*(0:(n-1))/n) ; sin(2*pi*(0:(n-1))/n)];
end
