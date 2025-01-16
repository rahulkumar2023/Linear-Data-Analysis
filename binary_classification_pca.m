function a5_00000000

    % Read the test data from a CSV file, standardize, and extract labels;
    % for the "college" data and Fisher's Iris data

    Xraw = csvread('collegenum.csv',1,1);
    [~, Xcoll] = pca(zscore(Xraw(:,2:end)), 'NumComponents', 2);
    ycoll = round(Xraw(:,1)>0);

    load fisheriris;
    Xiris = zscore(meas);
    yiris = ismember(species,'setosa');

    % Call the functions for the questions in the assignment
    a5q1(Xcoll, ycoll);
    a5q2(Xiris, yiris);
    
% END FUNCTION
end

function a5q1(Xmat, yvec)
% A5Q1(XMAT,YVEC) solves Question 1 for data in XMAT with
% binary labels YVEC
%
% INPUTS:
%         XMAT - MxN array, M observations of N variables
%         YVEC - Mx1 binary labels, interpreted as >=0 or <0
% OUTPUTS:
%         none

    % Augment the X matrix with a 1's vector
    Xaug = [Xmat ones(size(Xmat, 1), 1)];

    % Perceptron initialization and estimate of hyperplane
    eta = 0.001;
    [v_ann ix] = sepbinary(Xaug, yvec, eta);
    v_ann = v_ann/norm(v_ann);
    
    % Logistic regression estimate of hyperplane
    v_log = logreg(Xmat, yvec);
    v_log = v_log/norm(v_log);

    % Score the data using the hyperplane augmented vectors
    z_ann = Xaug*v_ann;
    z_log = Xaug*v_log;
    
    % Find the ROC curves
    [px_ann, py_ann, ~, auc_ann] = perfcurve(yvec, z_ann, +1);
    [px_log, py_log, ~, auc_log] = perfcurve(yvec, z_log, +1);
 
    % Initialize variables for accuracy and thresholds
    accuracyOFLOG = 0;
    accuracyOfANN = 0;
    thresholdOfLOG = 0;
    thresholdOfANN = 0;

% Compute accuracy for Logistic Regression and ANN
accuracyOFLOG = mean((z_log > thresholdOfLOG) == yvec);
accuracyOfANN = mean((z_ann > thresholdOfANN) == yvec);

% Display accuracies and AUCs
fprintf('Accuracy of Logistic Regression: %f\n', accuracyOFLOG);
fprintf('Accuracy of ANN: %f\n', accuracyOfANN);
fprintf('AUC of Logistic Regression: %f\n', auc_log);
fprintf('AUC of ANN: %f\n', auc_ann);

% Plot ROC curves for Logistic Regression and ANN
figure;
plot(px_log, py_log, 'b-', 'LineWidth', 3);
title('ROC Curve for Logistic Regression Model');
ylabel('True Positive Rate (TPR)');
xlabel('False Positive Rate (FPR)');
legend('Logistic Regression');

figure;
plot(px_ann, py_ann, 'r-', 'LineWidth', 3);
title('ROC Curve for ANN Model');
ylabel('True Positive Rate (TPR)');
xlabel('False Positive Rate (FPR)');
legend('Perceptron');

% Plot data points and separating hyperplane for Logistic Regression Model
figure; 
gscatter(Xmat(:,1), Xmat(:,2), yvec, 'rg', 'xo'); % Plot data points
hold on;
title('Logistic Regression Model - Dimensionally Reduced Data with Separating Hyperplane');
ylabel('Second Principal Component');
xlabel('First Principal Component');
legend('Class 0', 'Class 1');
plotline(v_log, 'k', 1); % Plot separating hyperplane
hold off;  

% Plot data points and separating hyperplane for ANN Model
figure; 
gscatter(Xmat(:,1), Xmat(:,2), yvec, 'rg', 'xo'); % Plot data points
hold on;
title('ANN Model - Dimensionally Reduced Data with Separating Hyperplane');
ylabel('Second Principal Component');
xlabel('First Principal Component');
legend('Class 0', 'Class 1');
plotline(v_ann, 'k', 1); % Plot separating hyperplane
hold off; 

% END FUNCTION
end

function [vector_final, iterations_used] = sepbinary(Xmat, yvec, eta_in)
% [V_FINAL,I_USED]=LINSEPLEARN(VINIT,ETA,XMAT,YVEC)
% uses the Percetron Algorithm to linearly separate training vectors
% INPUTS:
%         ZMAT    - Mx(N+1) augmented data matrix
%         YVEC    - Mx1 desired classes, 0 or 1
%         ETA     - optional, scalar learning rate, default is 1
% OUTPUTS:
%         V_FINAL - (N+1)-D new estimated weight vector
%         I_USED  - scalar number of iterations used
% ALGORITHM:
%         Vectorized form of perceptron gradient descent

   if exist('eta_in') && ~isempty(eta_in) && nargin >= 3
    % If eta_in exists, is not empty, and at least three input arguments are provided,
    % set the learning rate (eta) to the provided value.
    eta = eta_in;
else
    % Otherwise, set the default learning rate (eta) to 1.
    eta = 1;
end

% Initialize the weight vector as a vector of ones with the same size as the number of features in Xmat.
vector_est = ones(size(Xmat, 2), 1);

% Maximum number of iterations for the Perceptron algorithm.
iterationsmax = 10000;

% Loop a limited number of times to update the weight vector.
for iterations_used = 0:iterationsmax
    % Initialize the vector of errors (r_vector) to zeros.
    r_vector = zeros(size(yvec));
    % Variable to track if any data points are misclassified.
    miss = 0;
    % Assume that the final weight vector is the current estimate.
    vector_final = vector_est;

    % Compute the Perceptron update. 
    score = Xmat * vector_est; % Compute the dot product of the feature matrix and weight vector.
    r_vector = yvec - (score > 0); % Compute the vector of residuals by subtracting the model's binary predictions from the true labels.

    % Update the weight vector based on misclassified data points.
    for i = 1:size(Xmat, 1)
        if r_vector(i) ~= 0
            vector_est = vector_est + eta * r_vector(i) * Xmat(i, :)';
        end
    end

    % Check if any data points are misclassified.
    miss = norm(r_vector, 1) > 0;

    % If no data points are misclassified, exit the loop.
    if (miss == 0)
        vector_final = vector_est;
        break;
    end
end

% END FUNCTION
end

function a5q2(Xmat, yvec)
% A5Q2(XMAT,YVEC) solves Question 2 for data in XMAT with
% binary labels YVEC
%
% INPUTS:
%         XMAT - MxN array, M observations of N variables
%         YVEC - Mx1 binary labels, interpreted as ~=0 or ==0
% OUTPUTS:
%         none

    % Anonymous function: centering matrix of parameterized size
    Gmat =@(k) eye(k) - 1/k*ones(k,k);

    % Problem size
    [m, n] = size(Xmat);

    % Default projection of data
    Mgram = Xmat(:, 1:2);

    % Reduce data to Lmax-D; here, to 2D
    Lmax = 2;

    % Set an appropriate gamma for a Gaussian kernel
    sigma2 = 2*m;

    % Compute the centered MxM Gram matrix for the data
    Kmat = Gmat(m)*gramgauss(Xmat, sigma2)*Gmat(m);

  
    % Compute the eigenvectors and eigenvalues of the Gram matrix
[eigenVectors, eigenValues] = eig(Kmat);

% Extract eigenvalues from the diagonal of the eigenvalue matrix
eigenValues = diag(eigenValues); 

% Sort eigenvalues in descending order and rearrange corresponding eigenvectors
[~, sort_Index] = sort(eigenValues, 'descend');
eigenVectorsSorted = eigenVectors(:, sort_Index);

% Project the Gram matrix onto the principal components
Mgram = Kmat * eigenVectorsSorted(:, 1:Lmax); 

% Apply k-means clustering to the projected data
rng('default');
y_k2 = kmeans(Mgram, 2) - 1;

% Plot the results
% Plot Kernel PCA Projection using Original Labels
figure;
gscatter(Mgram(:,1), Mgram(:,2), yvec, ['r', 'b'], 'xo');
title('Kernel PCA Projection using Original Labels');
ylabel('Second Principal Component');
xlabel('First Principal Component');
legend({'Class 0', 'Class 1'}, 'Location', 'best');

% Plot Kernel PCA Projection using K-means clustering
figure;
gscatter(Mgram(:,1), Mgram(:,2), y_k2, ['m', 'c'], 'xo'); 
title('Kernel PCA Projection using K-means clustering');
ylabel('Second Principal Component');
xlabel('First Principal Component');
legend({'Cluster 1', 'Cluster 2'}, 'Location', 'best');

% END FUNCTION
end

function Kmat = gramgauss(Xmat, sigma2_in)
% K=GRAMGAUSS(X,SIGMA2)computes a Gram matrix for data in X
% using the Gaussian exponential exp(-1/sigma2*norm(X_i - X_j)^2)
%
% INPUTS:
%         X      - MxN data with M observations of N variables
%         sigma2 - optional scalar, default value is 1
% OUTPUTS:
%         K       NxN Gram matrix

% Check if sigma2_in is provided and the number of input arguments is at least 2
if ~isempty('sigma2_in') & (nargin>=2)
    sigma_2 = sigma2_in; % Use the provided sigma2_in value
else
    sigma_2 = 1; % Default value for sigma_2 if not provided
end

% Define a custom exponential function using sigma_2
my_Exp =@(urow,vmat) exp(-1/sigma_2*sum((vmat - urow).^2, 2));

% Compute the Gram matrix using the custom exponential function
Kmat = pdist2(Xmat, Xmat, my_Exp);

% END FUNCTION 
end

function waug = logreg(Xmat,yvec)
% WAUG=LOGREG(XMAT,YVEC) performs binary logistic regression on data
% matrix XMAT that has binary labels YVEC, using GLMFIT. The linear
% coefficients of the fit are in vector WAUG. Important note: the
% data XMAT are assumed to have no intercept term because these may be
% standardized data, but the logistic regression coefficients in WAUG
% will have an intercept term. The labels in YVEC are managed by
% >0 and ~>0, so either (-1,+1) convention or (0,1) convention in YVEC
% are acceptable.
%
% INPUTS:
%         XMAT - MxN array, of M observations in N variables
%         YVEC - Mx1 vector, binary labels
% OUTPUTS:
%         WAUG - (N+1)x1 vector, coefficients of logistic regression

    % Perform a circular shift of the GLMFIT coefficients so that
    % the final coefficient acts as an intercept term for XMAT
    
    warnstate = warning('query', 'last');
    warning('off');
    waug = circshift(glmfit(Xmat ,yvec>0, ...
        'binomial', 'link', 'probit'), -1);
    warning(warnstate);

    % END FUNCTION
end

function ph = plotline(vvec, color, lw, nv)
% PLOTLINE(VVEC,COLOR,LW,NV) plots a separating line
% into an existing figure
% INPUTS:
%        VVEC   - (M+1) augmented weight vector
%        COLOR  - character, color to use in the plot
%        LW   - optional scalar, line width for plotting symbols
%        NV   - optional logical, plot the normal vector
% OUTPUT:
%        PH   - plot handle for the current figure
% SIDE EFFECTS:
%        Plot into the current window. 

    % Set the line width
    if nargin >= 3 & ~isempty(lw)
        lwid = lw;
    else
        lwid = 2;
    end

    % Set the normal vector
    if nargin >= 4 & ~isempty(nv)
        do_normal = true;
    else
        do_normal = false;
    end

    % Current axis settings
    axin = axis();

    % Scale factor for the normal vector
    sval = 0.025*(axin(4) - axin(3));

    % Four corners of the current axis
    ll = [axin(1) ; axin(3)];
    lr = [axin(2) ; axin(3)];
    ul = [axin(1) ; axin(4)];
    ur = [axin(2) ; axin(4)];

    % Normal vector, direction vector, hyperplane scalar
    nlen = norm(vvec(1:2));
    uvec = vvec/nlen;
    nvec = uvec(1:2);
    dvec = [-uvec(2) ; uvec(1)];
    bval = uvec(3);

    % A point on the hyperplane
    pvec = -bval*nvec;

    % Projections of the axis corners on the separating line
    clist = dvec'*([ll lr ul ur] - pvec);
    cmin = min(clist);
    cmax = max(clist);

    % Start and end are outside the current plot axis, no problem
    pmin = pvec +cmin*dvec;
    pmax = pvec +cmax*dvec;

    % Create X and Y coordinates of a box for the current axis
    xbox = [axin(1) axin(2) axin(2) axin(1) axin(1)];
    ybox = [axin(3) axin(3) axin(4) axin(4) axin(3)];

    % Intersections of the line and the box
    [xi, yi] = polyxpoly([pmin(1) pmax(1)], [pmin(2) pmax(2)], xbox, ybox);

    % Point midway between the intersections
    pmid = [mean(xi) ; mean(yi)];

    % Range of the intersection line
    ilen = 0.5*norm([(max(xi) - min(xi)) ; (max(yi) - min(yi))]);

    % Plot the line according to the color specification
    hold on;
    if ischar(color)
        ph = plot([pmin(1) pmax(1)], [pmin(2) pmax(2)], ...
            [color '-'], 'LineWidth', lwid);
    else
        ph = plot([pmin(1) pmax(1)], [pmin(2) pmax(2)], ...
            'Color', color, 'LineStyle', '-', 'LineWidth', lwid);
    end
    if do_normal
        quiver(pmid(1), pmid(2), nvec(1)*ilen*sval, nvec(2)*ilen*sval, ...
            'Color', color, 'LineWidth', lwid, ...
            'MaxHeadSize', ilen/2, 'AutoScale', 'off');
    end
    hold off;
    
    % Remove this label from the legend, if any
    ch = get(gcf,'children');
    for ix=1:length(ch)
        if strcmp(ch(ix).Type, 'legend')
            ch(ix).String{end} = '';
        end
    end

% END FUNCTION
end
