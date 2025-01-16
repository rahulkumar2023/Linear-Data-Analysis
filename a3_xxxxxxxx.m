function a3_0000000

    % Read the data set from the CSv file provided
    dataset = csvread('wine.csv', 0,1);

    % Create a transpose of the matrix
    datatrans = dataset.'; 

    % Extract the y label identifier and the data matrix
    yvec = datatrans(:, 1);
    dataMat = datatrans(:, 2:end);
    
    % Compute Xmat's pair of columns with the lowest DB index
    [bestIndices, lowestDB] = findBestVar(dataMat, yvec);

    % Compute the PCA using SVD
    [score, PCA] = reduce(dataMat, yvec, false);

    % Standardized data usage
    [scorestandard, dbindexStandard] = reduce(dataMat, yvec, true);

    % Display the results
    datatable = {
    'Test', 'DB Index', 'Variable';
    'Data Columns', lowestDB, bestIndices;
    'Raw PCA Scores', PCA, 'N/A';
    'Standardized PCA', dbindexStandard, 'N/A';
    };

    % Convert the results to tabular form
    dbindtable = cell2table(datatable(2:end, :), 'VariableNames', datatable(1, :));
    disp(dbindtable);

    % Plot the first figure for the best pair of indexes or columns
    figure;
    gscatter(dataMat(:,1),dataMat(:,7),yvec)
    title('Best pair of indexes');
    xlabel('Column 1');
    ylabel('Column 7');

    % Plot the second figure for the raw PCA score and its corresponding principal
    % components
    figure;
    gscatter(score(:,1),score(:,2),yvec)
    title('Raw PCA');
    xlabel('Principal Component 1');
    ylabel('Principal Component 2');

    % Plot the third figure for the standardized PCA score and its corresponding principal
    % components
    figure;
    gscatter(scorestandard(:,1),scorestandard(:,2),yvec)
    title('Standardized PCA');
    xlabel('Principal Component 1');
    ylabel('Principal Component 2');
end


function [bestIndex, lowestDB] = findBestVar(dataMat, yvec)

    % Initialize the best pair of columns and the lowest DB index that
    % corresponds to those columns
    bestpair = [];
    minDB = inf;
    
    % Iterate through all pairs of unique values
    for i = 1:size(dataMat, 2)
        for j = i+1:size(dataMat, 2)

            % Create a design matrix with each pair of values
            xmat = dataMat(:, [i, j]);

            % Compute the DB index for each pair of values
            currentDBindex = dbindex(xmat, yvec);

            % Check if the current pair of values has a lower DB index than
            % the lowest DB index found so far
            if currentDBindex < minDB
                minDB = currentDBindex;
                bestpair = [i, j];
            end
        end
    end

    % Update the lowest DB index variable and bestIndex variable with the
    % values found
    lowestDB = minDB;
    bestIndex = bestpair;
end


function [score, DBInd] = reduce(dataMat, yvec, standard)

    % Conditional code in the case of data being required for purposes of
    % standardization
    if standard
        dataMat = zscore(dataMat);
    end

    % Get the zero mean of the data
    zeromean = dataMat - mean(dataMat);

    % Find SVD
    [U, S, V] = svd(zeromean);

    % Use SVD for PCA score
    score = zeromean * V(:,1:2);

    % Compute the DB index based on the PCA score
    DBInd = dbindex(score, yvec);
end


function score = dbindex(Xmat, lvec)
% SCORE=DBINDEX(XMAT,LVEC) computes the Davies-Bouldin index
% for a design matrix XMAT by using the values in LVEC as labels.
% The calculation implements a formula in their journal article.
%
% INPUTS:
%        XMAT  - MxN design matrix, each row is an observation and
%                each column is a variable
%        LVEC  - Mx1 label vector, each entry is an observation label
% OUTPUT:
%        SCORE - non-negative scalar, smaller is "better" separation

    % Anonymous function for Euclidean norm of observations
    rownorm = @(xmat) sqrt(sum(xmat.^2, 2));

    % Problem: unique labels and how many there are
    kset = unique(lvec);
    k = length(kset);

    % Loop over all indexes and accumulate the DB score of each label
    % gi is the label centroid
    % mi is the mean distance from the centroid
    % Di contains the distance ratios between IX and each other label
    D = [];
    for ix = 1:k
        Xi = Xmat(lvec==kset(ix), :);
        gi = mean(Xi);
        mi = mean(rownorm(Xi - gi));
        Di = [];
        for jx = 1:k
            if jx~=ix
                Xj = Xmat(lvec==kset(jx), :);
                gj = mean(Xj);
                mj = mean(rownorm(Xj - gj));
                Di(end+1) = (mi + mj)/norm(gi - gj);
            end
        end
        D(end+1) = max(Di);
    end

    % DB score is the mean of the scores of the labels
    score = mean(D);
end
