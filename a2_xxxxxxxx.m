function [rmsvars lowIndexPositive lowIndexNegative rmstrain rmstest] = a2_00000000
% [RMSVARS LOWNDX RMSTRAIN RMSTEST]=A3 finds the RMS errors of
% linear regression of the data in the associated CSV file. The
% individual RMS errors are returned in RMSVARS and the index of the
% smallest RMS error is returned in LOWNDX. For the variable that
% best explains the dependent variable, a 5-fold cross validation is
% computed. The RMS errors for the training of each fold are returned
% in RMSTEST and the RMS errors for the testing of each fold are
% returned in RMSTEST.
%
% INPUTS:
%         none
% OUTPUTS:
%         RMSVARS  - 1xN array of RMS errors of linear regression
%         LOWINDEXPOSITIVE - integer scalar, index into RMSVALS
%         LOWINDEXNEGATIVE - integer scalar, index into RMSVALS
%         RMSTRAIN - 1x5 array of RMS errors for 5-fold training
%         RMSTEST  - 1x5 array of RMS errors for 5-fold testing

    [rmsvars lowIndexPositive lowIndexNegative] = a2q1
    [rmstrain rmstest] = a2q2(lowIndexPositive)

end

function [rmsvars, lowIndexPositive, lowIndexNegative] = a2q1
    % [RMSVARS LOWNDX]=A2Q1 finds the RMS errors of
    % linear regression of the data in the CSV file. The
    % individual RMS errors are returned in RMSVARS and the index of the
    % smallest RMS error is returned in LOWNDX.

    % Read the test data from a CSV file and find the size of the data
    [fragilityVector, dataMatrix, countries, ageMatrix] = fragilitydata;
    [m, n] = size(dataMatrix);

    % Compute the RMS errors for linear regression
    posmin = inf;
    negmin = inf;
    rmsvars = zeros(1, n);

    for i = 1:n
        % Ones vector for this data
        onesVector = ones(size(fragilityVector, 1), 1);

        % Form the design matrix
        designMat = [dataMatrix(:, i), onesVector];

        % Solve the linear regression
        linearreg = designMat \ fragilityVector;

        % Find RMS errors
        rmsval = rms(fragilityVector - designMat * linearreg);

        % Update minimum RMS and corresponding index
        if linearreg(1) > 0
            if rmsval < posmin
                posmin = rmsval;
                lowIndexPositive = i;
            end
        else
            if rmsval < negmin
                negmin = rmsval;
                lowIndexNegative = i;
            end
        end

        rmsvars(i) = rmsval;
    end

    % Plot the result
    ageLo = ageMatrix(1, 2); % Where the age range starts
    ageHi = ageMatrix(2, 2); % Where the age range ends
    ageString = sprintf('Males in range %d:%d', ageLo, ageHi);
    
    % Sample plot
    figure;
    plot(dataMatrix(:, 2), fragilityVector, '.', 'MarkerSize', 10);
    xlabel('Proportion of Male Population');
    ylabel('Fragility Index');
    title(ageString);

    % Add linear fit to the plot
    designMat_2 = [dataMatrix(:, 2), onesVector];
    widthfit = designMat_2 \ fragilityVector;
    
    axisV = axis();
    xVals = axisV(1:2); % X for the left and right sides of the plot
    yVals = widthfit(1) * xVals + widthfit(2); % Slope/intercept computation
    hold on;
    plot(xVals, yVals, 'k-');
    hold off;

    function [rmsvars, lowIndexPositive, lowIndexNegative] = a2q1(lowIndexNegative)
    % ... (existing code)

    % Update minimum RMS and corresponding index
    if linearreg(1) > 0
        if rmsval < posmin
            posmin = rmsval;
            lowIndexPositive = i;
        end
    else
        if rmsval < negmin
            negmin = rmsval;
            lowIndexNegative = i;
        end
    end

    rmsvars(i) = rmsval;
    end

    % Plot the result for the lowest negative index
    plotResult(dataMatrix(:, lowIndexNegative), fragilityVector, ageMatrix(:, lowIndexNegative), onesVector);
end

function plotResult(xData, yData, ageRange, onesVector)
    ageLo = ageRange(1); % Where the age range starts
    ageHi = ageRange(2); % Where the age range ends
    ageString = sprintf('Males in range %d:%d', ageLo, ageHi);

    % Sample plot
    figure;
    plot(xData, yData, '.', 'MarkerSize', 10);
    xlabel('Proportion of Male Population');
    ylabel('Fragility Index');
    title(ageString);

    % Add linear fit to the plot
    designMat = [xData, onesVector];
    widthfit = designMat \ yData;

    axisV = axis();
    xVals = axisV(1:2); % X for the left and right sides of the plot
    yVals = widthfit(1) * xVals + widthfit(2); % Slope/intercept computation
    hold on;
    plot(xVals, yVals, 'k-');
    hold off;
end
    

function [rmstrain, rmstest] = a2q2(lowndx)
    % [RMSTRAIN RMSTEST]=A3Q2(LOWNDX) finds the RMS errors of 5-fold
    % cross-validation for the variable LOWNDX of the data in the CSV file.
    % The RMS errors for the training of each fold are returned
    % in RMSTEST and the RMS errors for the testing of each fold are
    % returned in RMSTEST.

    % Read the test data from a CSV file and find the size of the data
    [fragilityVector, dataMatrix, countries, ageMatrix] = fragilitydata;

    % Assign the variables from the dataset
    rng('default'); % makes sure to get the same results each time (consistent seed)
    randidx = randperm(size(fragilityVector, 1)); % random permutation of rows

    lowdata = dataMatrix(:, lowndx);
    yvec = fragilityVector(randidx, :); % rearrange order of rows in yvec;
    Xmat = lowdata(randidx, :); % rearrange order of rows in Xmat;

    k_in = 5;

    % Compute the RMS errors of 5-fold cross-validation
    [rmstrain, rmstest] = mykfold(Xmat, yvec, k_in);
end

function [rmstrain, rmstest] = mykfold(Xmat, yvec, k_in)
    % [RMSTRAIN,RMSTEST]=MYKFOLD(XMAT,yvec,K) performs a k-fold validation
    % of the least-squares linear fit of yvec to XMAT. If K is omitted,
    % the default is 5.

    % Problem size
    M = size(Xmat, 1);

    % Set the number of folds; must be 1 < k < M
    if nargin >= 3 && ~isempty(k_in)
        k = max(min(round(k_in), M - 1), 2);
    else
        k = 5;
    end

    % Determine the number of rows per experiment
    numInFold = round(size(Xmat, 1) / k);

    % Initialize the return variables
    rmstrain = zeros(1, k);
    rmstest = zeros(1, k);

    starttest = 1;
    endtest = numInFold;

    % Process each fold
    for ix = 1:k
        testindices = false(size(Xmat, 1), 1);
        testindices(starttest:endtest) = true;
        trainindices = ~testindices;

        % Extract data for train and test
        xtest = Xmat(testindices, :);
        ytest = yvec(testindices, :);
        xtrain = Xmat(trainindices, :);
        ytrain = yvec(trainindices, :);

        % Compute "wvec" for the training data
        wvec = xtrain \ ytrain;

        % Compute RMS errors for training and testing
        rmstrain(ix) = rms(ytrain - xtrain * wvec);
        rmstest(ix) = rms(ytest - xtest * wvec);

        starttest = endtest + 1;
        if starttest + numInFold - 1 > size(Xmat, 1)
            endtest = size(Xmat, 1);
        else
            endtest = starttest + numInFold - 1;
        end
    end
end
function [fragilityVector, dataMatrix, countries, ageMatrix] = ...
    fragilitydata
% [fragilityVector,dataMatrix,countries,ageMatrix]=fragilitydata
% loads and separates the 2013 fragility data for assessed countries
% N.B. These are proportions of male populations; other data
% are not loaded here. UN population estimates were used
%
% INPUTS:
%         none
% OUTPUTS:
%         fragilityVector - Mx1 vector of fragility "index" values
%         dataMatrix      - MxN matrix, M countries and N age groups
%         countries       - Mx1 cell array, strings for country names
%         ageMatrix       - 2xN matrix, start end end of each age group

    % Load the table from a known CSV file
    mt = readtable('fragility2013male.csv');

    % Extract the values of the fragility "index" for each country
    fragilityVector = table2array(mt(:,2));

    % Extract and normalize data for male populations
    dataRaw = table2array(mt(:,3:end));
    dataMatrix = dataRaw./sum(dataRaw, 2);

    % Extract the country names
    countries = table2array(mt(:,1));

    allNames = mt.Properties.VariableNames;
    ages = allNames(3:end);
    N = length(ages);

    ageMatrix = zeros(2, N);
    for jx = 1:N
        ageString = cell2mat(ages(jx));
        lowStart = strfind(ageString, 'm');
        lowEnd = strfind(ageString, '_');
        ageMatrix(1, jx) = str2num(ageString((lowStart+1):(lowEnd-1)));
        endValue = str2num(ageString((lowEnd+1):end));
        if ~isempty(endValue)
            ageMatrix(2, jx) = endValue;
        else
            ageMatrix(2, jx) = inf;
        end
    end
end