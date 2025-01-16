function a4_00000000
% Function for CISC271, Winter 2022, Assignment #4

    % Read the test data from a CSV file
    dmrisk = csvread('dmrisk.csv',1,0);

    % Columns for the data and labels; DM is diabetes, OB is obesity
    jDM = 17;
    jOB = 16;

    % Extract the data matrices and labels
    XDM = dmrisk(:, (1:size(dmrisk,2))~=jDM);
    yDM = dmrisk(:,jDM);
    XOB = dmrisk(:, (1:size(dmrisk,2))~=jOB);
    yOB = dmrisk(:,jOB);

    % Reduce the dimensionality to 2D using PCA
    [~,rDM] = pca(zscore(XDM), 'NumComponents', 2);
    [~,rOB] = pca(zscore(XOB), 'NumComponents', 2);

    % Find the LDA vectors and scores for each data set
    [qDM zDM qOB zOB] = a4q1(rDM, yDM, rOB, yOB);

    % %
    % % STUDENT CODE GOES HERE: PLOT RELEVANT DATA
    % %
    
    
    % Create a new figure for the plot
    figure;

    % Define y-coordinates based on class labels for DM data
    ycoordinatesofDM = yDM;
    ycoordinatesofDM(yDM == 1) = 1; % Assign y=1 for Class 1
    ycoordinatesofDM(yDM ~= 1) = 2; % Assign y=2 for Class 2

    % Scatter plot of LDA scores for DM data with distinct class representation
    gscatter(zDM, ycoordinatesofDM, yDM, 'rg', 'xo');
    title('LDA scores of Diabetes'); % Set the title of the plot
    ylabel('Type of Class'); % Label for the y-axis
    xlabel('LDA Score'); % Label for the x-axis
    xlim([-3, 6]); ylim([0 5]); % Set limits for the x and y axes

    % Set Y-axis ticks and labels for better interpretation
    set(gca, 'YTick', [1, 2], 'YTickLabel', {'Positive', 'Negative'});

    % Add a grid to the plot for better visualization
    grid on;

    % Create a new figure for the plot
    figure;

    % Define y-coordinates based on class labels for Obesity data
    ycoordinatesofOB = yOB;
    ycoordinatesofOB(yOB == 1) = 1; % Assign y=1 for Class 1
    ycoordinatesofOB(yOB ~= 1) = 2; % Assign y=2 for Class 2

    % Scatter plot of LDA scores for Obesity data with distinct class representation
    gscatter(zOB, ycoordinatesofOB, yOB, 'rg', 'xo');
    title('LDA Scores of Obesity'); % Set the title of the plot
    ylabel('Type of Class'); % Label for the y-axis
    xlabel('LDA Score'); % Label for the x-axis
    xlim([-3, 6]); ylim([0 5]); % Set limits for the x and y axes

    % Set Y-axis ticks and labels for better interpretation
    set(gca, 'YTick', [1, 2], 'YTickLabel', {'Positive', 'Negative'});

    % Add a grid to the plot for better visualization
    grid on;
    
    % Plotting 2D data with LDA scores for DM
    figure;
    gscatter(rDM(:,1), rDM(:,2), yDM, 'rg', 'xo');
    title('2D DM data'); % Set the title of the plot
    ylabel('Principal Component 2'); % Label for the y-axis
    xlabel('Principal Component 1'); % Label for the x-axis
    legend('Positive', 'Negative'); % Show legend for class representation
    grid on; % Add a grid to the plot for better visualization

    % Plotting 2D data with LDA scores for OB
    figure;
    gscatter(rOB(:,1), rOB(:,2), yOB, 'rg', 'xo');
    title('2D OB data'); % Set the title of the plot
    ylabel('Principal Component 2'); % Label for the y-axis
    xlabel('Principal Component 1'); % Label for the x-axis
    legend('Positive', 'Negative'); % Show legend for class representation
    grid on; % Add a grid to the plot for better visualization

    % Compute ROC curve, AUC, and optimal threshold for DM data
    [FPRD, TPRD, AUCD, OPTD] = roccurve(yDM, zDM);

    % Compute ROC curve, AUC, and optimal threshold for OB data
    [FPRO, TPRO, AUCO, OPTO] = roccurve(yOB, zOB);

    % Display optimal thresholds for DM and OB data
    OPTD
    OPTO

    % Compute confusion matrices for DM and OB data using optimal thresholds
    confusionMATDM = confmat(yDM, zDM, OPTD);
    confusionMATOB = confmat(yOB, zOB, OPTO);

    % Display AUC and optimal confusion matrix for DM data
    disp(['AUC for Diabetes Data: ', num2str(AUCD)]);
    disp('Optimal Confusion Matrix for Diabetes Data:');
    disp(confusionMATDM);

    % Display AUC and optimal confusion matrix for OB data
    disp(['AUC for Obesity Data: ', num2str(AUCO)]);
    disp('Optimal Confusion Matrix for Obesity Data:');
    disp(confusionMATOB);

    % Plot ROC curves for DM and OB data
    figure;
    plot(FPRD, TPRD, 'r', 'LineWidth', 3, 'DisplayName', 'DM ROC Curve');
    hold on;
    plot(FPRO, TPRO, 'g', 'LineWidth', 3, 'DisplayName', 'OB ROC Curve');
    title('Receiver Operating Characteristic (ROC) Curves'); % Set the title of the plot
    ylabel('True Positive Rate (TPR)'); % Label for the y-axis
    xlabel('False Positive Rate (FPR)'); % Label for the x-axis
    legend('show'); % Show legend for different ROC curves
end

function [q1, z1, q2, z2] = a4q1(Xmat1, yvec1, Xmat2, yvec2)
% [Q1 Z1 Q2 Z2]=A4Q1(X1,Y1,X2,Y2) computes an LDA axis and a
% score vector for X1 with Y1, and for X2 with Y2.
%
% INPUTS:
%         X1 - MxN data, M observations of N variables
%         Y1 - Mx1 labels, +/- computed as ==/~= 1
%         X2 - MxN data, M observations of N variables
%         Y2 - Mx1 labels, +/- computed as ==/~= 1
% OUTPUTS:
%         Q1 - Nx1 vector, LDA axis of data set #1
%         Z1 - Mx1 vector, scores of data set #1
%         Q2 - Nx1 vector, LDA axis of data set #2
%         Z2 - Mx1 vector, scores of data set #2

    q1 = [];
    z1 = [];
    q2 = [];
    z2 = [];
    
    % Compute the LDA axis for each data set
    q1 = lda2class(Xmat1(yvec1==1,:), Xmat1(yvec1~=1, :));
    q2 = lda2class(Xmat2(yvec2==1,:), Xmat2(yvec2~=1, :));
   
    % %
    % % STUDENT CODE GOES HERE: COMPUTE SCORES USING LDA AXES
    % 
   
    % Compute the mean vectors for each dataset
    meanDS1 = mean(Xmat1, 1);
    meanDS2 = mean(Xmat2, 1);

    % Center the data by subtracting the mean vector
    % Multiply by q1 (LDA axis) to obtain the scores for dataset 1
    z1 = (Xmat1 - ones(size(Xmat1, 1), 1) * meanDS1) * q1;

    % Center the data by subtracting the mean vector
    % Multiply by q2 (LDA axis) to obtain the scores for dataset 2
    z2 = (Xmat2 - ones(size(Xmat2, 1), 1) * meanDS2) * q2;

% END OF FUNCTION
end

function qvec = lda2class(X1, X2)
% QVEC=LDA2(X1,X2) finds Fisher's linear discriminant axis QVEC
% for data in X1 and X2.  The data are assumed to be sufficiently
% independent that the within-label scatter matrix is full rank.
%
% INPUTS:
%         X1   - M1xN data with M1 observations of N variables
%         X2   - M2xN data with M2 observations of N variables
% OUTPUTS:
%         qvec - Nx1 unit direction of maximum separation
    
% Combine the data from both datasets
A = [X1; X2];

% Compute the mean vector for the combined dataset
DSMeanA = mean(A);

% Separate the datasets
A1 = X1;
DSMeanA1 = mean(A1);

A2 = X2;
DSMeanA2 = mean(A2);

% Center the data for each dataset individually
M1 = -ones(size(A1, 1), 1) * DSMeanA1 + A1;
M2 = -ones(size(A2, 1), 1) * DSMeanA2 + A2;

% Compute the within-class scatter matrices
S1 = M1' * M1;
S2 = M2' * M2;
Sw = S1 + S2;

% Compute the between-class scatter matrix
T = [DSMeanA1 - DSMeanA; DSMeanA2 - DSMeanA];
Sb = T' * T;

% Compute the Fisher's linear discriminant axis
[eigvectors, eigvalues] = eig(Sw \ Sb);
[~, maxIndex] = max(diag(eigvalues));
fDiscriminant = eigvectors(:, maxIndex);

% Ensure the direction of the discriminant is correct
direction = DSMeanA1 - DSMeanA2;
if (direction * fDiscriminant < 0)
    fDiscriminant = -fDiscriminant;
end

% The final discriminant vector (LDA axis)
qvec = fDiscriminant;
end

function [FPR, TPR, auc, bopt] = roccurve(yvec_in, zvec_in)
    % [FPR TPR AUC BOPT]=ROCCURVE(YVEC,ZVEC) computes the
    % ROC curve and related values for labels YVEC and scores ZVEC.
    % Unique scores are used as thresholds for binary classification.
    %
    % INPUTS:
    %         YVEC - Mx1 labels, +/- computed as ==/~= 1
    %         ZVEC - Mx1 scores, real numbers
    % OUTPUTS:
    %         FPR  - Kx1 vector of False Positive Rate values
    %         TPR  - Kx1 vector of True Positive Rate values
    %         AUC  - scalar, Area Under Curve of ROC determined by TPR and FPR
    %         BOPT - scalar, optimal threshold for accuracy

    
% Sort the scores and permute the labels accordingly
[zvec, zndx] = sort(zvec_in);
yvec = yvec_in(zndx);

% Sort and find a unique subset of the scores
bvec = unique(zvec);
bm = numel(bvec);

% Initialize vectors for True Positive Rate, False Positive Rate, accuracy, and optimal threshold
TPR = zeros(bm, 1);
FPR = zeros(bm, 1);
acc = zeros(bm, 1);
bopt = bvec(1); % Initialize optimal threshold with the first unique score

% Iterate over each unique threshold value
for jx = 1:bm
    % Compute the confusion matrix for the current threshold
    confusionmatrix = confmat(yvec, zvec, bvec(jx));
    
    % Extract values from the confusion matrix
    TP = confusionmatrix(1, 1);
    TN = confusionmatrix(2, 2);
    FP = confusionmatrix(2, 1);
    FN = confusionmatrix(1, 2);

    % Compute True Positive Rate (Sensitivity) and False Positive Rate
    TPR(jx) = TP / (TP + FN);
    TNR = TN / (TN + FP);
    FPR(jx) = FP / (FP + TN);

    % Compute False Negative Rate and accuracy
    FNR = FN / (TP + FN);
    acc(jx) = (TPR(jx) * TNR) - (FPR(jx) * FNR);
end

% Find the index of the optimal threshold with maximum accuracy
[~, idxOpt] = max(acc);
bopt = bvec(idxOpt);

% Ensure that the rates are sorted for correct plotting
TPR = sort(TPR);
FPR = sort(FPR);

% Compute AUC for the ROC curve
auc = aucofroc(FPR, TPR);

end
    
function cmat = confmat(yvec, zvec, theta)
% CMAT=CONFMAT(YVEC,ZVEC,THETA) finds the confusion matrix CMAT for labels
% YVEC from scores ZVEC and a threshold THETA. YVEC is assumed to be +1/-1
% and each entry of ZVEC is scored as -1 if <THETA and +1 otherwise. CMAT
% is returned as [TP FN ; FP TN]
%
% INPUTS:
%         YVEC  - Mx1 values, +/- computed as ==/~= 1
%         ZVEC  - Mx1 scores, real numbers
%         THETA - threshold real-valued scalar
% OUTPUTS:
%         CMAT  - 2x2 confusion matrix; rows are +/- labels,
%                 columns are +/- classifications

  
% Quantize the scores based on the threshold
qvec = sign((zvec >= theta) - 0.5);

% Replace zero values in qvec with 1
qvec(qvec == 0) = 1;

% Compute True Positives (TP), True Negatives (TN), False Positives (FP),
% and False Negatives (FN) based on the quantization
TP = sum((yvec == 1) & (qvec == 1)); % True Positives
TN = sum((yvec == -1) & (qvec == -1)); % True Negatives
FP = sum((yvec == -1) & (qvec == 1)); % False Positives
FN = sum((yvec == 1) & (qvec == -1)); % False Negatives

% Create the confusion matrix
cmat = [TP FN; FP TN];
end

function auc = aucofroc(fpr, tpr)
% AUC=AUCOFROC(TPR,FPR) finds the Area Under Curve of the
% ROC curve specified by the TPR, True Positive Rate, and
% the FPR, False Positive Rate.
%
% INPUTS:
%         TPR - Kx1 vector, rate for underlying score threshold 
%         FPR - Kx1 vector, rate for underlying score threshold 
% OUTPUTS:
%         AUC - integral, from Trapezoidal Rule on [0,0] to [1,1]

    [X undx] = sort(reshape(fpr, 1, numel(fpr)));
    Y = sort(reshape(tpr(undx), 1, numel(undx)));
    auc = abs(trapz([0 X 1] , [0 Y 1]));
end
