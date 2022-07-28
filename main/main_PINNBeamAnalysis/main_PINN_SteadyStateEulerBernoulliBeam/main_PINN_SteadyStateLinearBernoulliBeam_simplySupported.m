%% Preamble
clc, clear;

%% Includes
addpath("../auxiliary/");

%% Define the problem parameters

propStr.q0 = 1e2; % distributed load
propStr.E = 1e7; % Young's modulus
propStr.b = .1; % width
propStr.h = .1; % height
propStr.I = propStr.b*(propStr.h^3)/12; % moment of inertia

% Amplitude at the beam's ends
w0 = 0;

% Computational domain
propGeom.X0 = 0;
propGeom.XL = 1;
XComp = [propGeom.X0 propGeom.XL];

%% Define analytical solution based on the given parameters
syms w_analytical(X) q E I X0 XL

% Method for obtaining the analytical parabolic solution for the benchmark
% problem at hand (comment out the following lines to employ it):
%
% syms a b c d e E I X X0 XL q wmax
% eqn1 = a*X0^4 + b*X0^3 + c*X0^2 + d*X0 + e == 0;
% eqn2 = a*XL^4 + b*XL^3 + c*XL^2 + d*XL + e == 0;
% eqn3 = 12*a*X0^2 + 6*b*X0 + 2*c == 0;
% eqn4 = 12*a*XL^2 + 6*b*XL + 2*c == 0;
% eqn5 = a*((X0 + XL)/2)^4 + b*((X0 + XL)/2)^3 + c*((X0 + XL)/2)^2 + d*(X0 + XL)/2 + e - wmax == 0;
% eqns = [eqn1, eqn2, eqn3, eqn4, eqn5];
% S = solve(eqns, [a b c d e]);
% expr = S.a*X^4 + S.b*X^3 + S.c*X^2 + S.d*X + S.e;
% wmax = 0.0156
% expr = subs(expr, wmax);
% expr = simplify(expr);
%
% expr =
% -(156*(X - X0)*(X - XL)*(- X^2 + X*X0 + X*XL + X0^2 - 3*X0*XL + XL^2))/(3125*(X0 - XL)^4)
w_analytical(X) = -(156*(X - X0)*(X - XL)*(- X^2 + X*X0 + X*XL + X0^2 - 3*X0*XL + XL^2))/(3125*(X0 - XL)^4);
w_analytical(X) = subs(w_analytical(X), {q, E, I, X0, XL}, ...
    {propStr.q0, propStr.E, propStr.I, propGeom.X0, propGeom.XL});

%% Generate the training data

% points over which to enforce the boundary conditions by means of the loss
% of the neural network
numBC = 2;

% boundary conditions
XBC = XComp;
WBC = ones(1, numBC)*w0;

%% Create some random points between the computational interval
numInternColl = 1e2; % 1e3
pointSet = sobolset(1);
points = net(pointSet, numInternColl);
X = XComp(1, 1) + points'*(XComp(1, 2) - XComp(1, 1));
% X = linspace(0.1, 1.1, 10);

%% Deep learning model

% Number of layers
numLayers = 2; % 3

% Number of neurons per layer
numNeurons = 5; % 10

% initialize the weights and biases for the first fully connected operation
% The input layer has 1 neuron and numNeurons connections to the next layer
sz = [numNeurons 1];
parameters.fc1_Weights = initializeHe(sz, 1, 'double');
parameters.fc1_Bias = initializeZeros([numNeurons 1],'double');

% Initialize the weights and biases for the hidden fully connected 
% operations. Note that each hidden layer has numNeurons x numNeurons 
% connections
for layerNumber = 2:numLayers - 1
    name = "fc" + layerNumber;

    sz = [numNeurons numNeurons];
    numIn = numNeurons;
    parameters.(name + "_Weights") = initializeHe(sz, numIn,'double');
    parameters.(name + "_Bias") = initializeZeros([numNeurons 1],'double');
end

% Initialize the final fully connected operation. The output layer has only 
% one neuron and returns the predicted solution w(x)
sz = [1 numNeurons];
numIn = numNeurons;
parameters.("fc" + numLayers + "_Weights") = initializeHe(sz, numIn,'double');
parameters.("fc" + numLayers + "_Bias") = initializeZeros([1 1],'double');

%% Specify the optimization options
opts = optimoptions('fmincon', ... % 'fmincon', 'fminunc' (R2022a)
    'HessianApproximation', 'lbfgs', ... % {'lbfgs', 5} L-BFGS quasi-Newton method
    'MaxIterations', 2000, ... % 7500
    'MaxFunctionEvaluations', 2000, ... % 7500
    'OptimalityTolerance', 1e-12, ... % 1e-5 for the example from Niccolo
    'SpecifyObjectiveGradient', true, ...
    'Display', 'iter'); % 'iter', 'none'

%% Train the network using fmincon

% learning parameters
propLearning.rhoR = 1e-2;
propLearning.numR = numInternColl;
propLearning.rhoB = 1;
propLearning.numB = numBC;

% extract data and names from the parameters
[parametersV,parameterNames,parameterSizes] = parameterStructToVector...
    (parameters);
parametersV = extractdata(parametersV);

% Convert the training data to dlarray objects with format 'CB' (channel, batch).
dlX = dlarray(X,'CB');
dlXBC = dlarray(XBC,'CB');
dlWBC = dlarray(WBC,'CB');

% Create a function handle with one input that defines the objective function.
objFun = @(parameters) objectiveFunction(parameters, dlX, dlXBC, dlWBC, ...
    parameterNames, parameterSizes, w_analytical, propStr, propGeom, propLearning);

% Update the learnable parameters using the fmincon function
% figure(1)
parametersV = fmincon(objFun, parametersV, [], [], [], [], [], [], [], ...
    opts);
% parametersV = fminunc(objFun, parametersV, opts);
% parametersV = ga(objFun, parametersV, [], [], [], [], [], [], [], ...
%     opts);
% parametersV = simulannealbnd(objFun, parametersV);

% For prediction convert the vector of parameters to a structure
parameters = parameterVectorToStruct(parametersV, ...
    parameterNames, parameterSizes);

%% Evaluate the model's accuracy

% create a set of collocation points in the domain to predict the solution
XTest = linspace(XComp(1, 1), XComp(1, 2), 100);
dlXTest = dlarray(XTest, 'CB');

% predict the solution using the trained model
dlWTest = model(parameters, dlXTest);
WTest = extractdata(dlWTest);

% Plot the solution against the expected one from isogeometric analysis
figure(2)
plot(XTest, WTest, '-black', 'Linewidth', 2);
hold on
fplot(w_analytical, [propGeom.X0, propGeom.XL],'--r', 'Linewidth',2);
hold off
xlabel('X');
ylabel('X_0 + w');
title("Beam's deformation");
legend('PINN', 'IGA');
drawnow

% This is the maximum deflection at the middle of the beam for this loading
% and it should be upwards
% 
% ans =
% 
%     0.0156

%% Auxiliary functions

%% Define the objective function that optimizes for the weights and biases
function [loss, gradientsV] = objectiveFunction...
    (parametersV, dlX, dlXBC, dlWBC, parameterNames, parameterSizes, ...
    w_analytical, propStr, propGeom, propLearning)

    % Convert parameters to structure of dlarray objects
    parametersV = dlarray(parametersV);
    parameters = parameterVectorToStruct(parametersV, parameterNames, ...
        parameterSizes);
    
    % Evaluate model gradients and loss
    [gradients, loss] = dlfeval ...
        (@modelGradients, parameters, dlX, dlXBC, dlWBC, w_analytical, ...
        propStr, propGeom, propLearning);
    
    % Return loss and gradients for fmincon
    gradientsV = parameterStructToVector(gradients);
    gradientsV = extractdata(gradientsV);
    loss = extractdata(loss);
end

%% Function that returns the loss and its gradient with respect to the weights and biases
function [gradients,loss] = modelGradients ...
    (parameters, dlX, dlXBC, dlWBC, w_analytical, propStr, propGeom, propLearning)

    % Make predictions with the initial conditions
    W = model(parameters, dlX);

    % Plot the intermediate solution (Debugging)
    [X_i, order] = sort(extractdata(dlX));
    W_i = extractdata(W);
    plot(X_i, W_i(order), '-black', 'Linewidth',2);
    hold on
    fplot(w_analytical, [propGeom.X0, propGeom.XL],'--r', 'Linewidth',2);
    hold off
    xlabel('X');
    ylabel('X_0 + w');
    title("Beam's deformation");
    legend('PINN', 'IGA');
    drawnow
    
    % Calculate the derivatives with respect to X
    Wx = dlgradient(sum(W,'all'), dlX, 'EnableHigherDerivatives', true);
    Wxx = dlgradient(sum(Wx,'all'), dlX, 'EnableHigherDerivatives', true);
    Wxxx = dlgradient(sum(Wxx,'all'), dlX, 'EnableHigherDerivatives', true);
    Wxxxx = dlgradient(sum(Wxxx,'all'), dlX, 'EnableHigherDerivatives', true);
    
    % Calculate lossF. Enforce the Euler-Bernoulli equation
    f = propStr.E*propStr.I*Wxxxx - propStr.q0;
    zeroTarget = zeros(size(f), 'like', f);
    lossF = mse(f, zeroTarget);
    
    % Calculate lossU. Enforce initial and boundary conditions.
    dlU0Pred = model(parameters, dlXBC);
    lossU = mse(dlU0Pred, dlWBC);

    % Calculate the second derivatives of the boundary conditions with 
    % respect to X
    dlWBCx = dlgradient(sum(dlU0Pred,'all'), dlXBC, 'EnableHigherDerivatives', true);
    dlWBCxx = dlgradient(sum(dlWBCx,'all'), dlXBC, 'EnableHigherDerivatives', true);

    % Enforce the second derivative of the displacement field to zero
    zeroTargetDDwDDX = zeros(size(dlWBCxx), 'like', f);
    lossDDwDDX = mse(dlWBCxx, zeroTargetDDwDDX);
    
    % Combine losses into the general loss function
    loss = propLearning.rhoR*(1/propLearning.numR)*lossF + ...
        propLearning.rhoB*(1/propLearning.numB)*(lossU + lossDDwDDX);
    
    % Calculate gradients with respect to the learnable parameters.
    gradients = dlgradient(loss, parameters);
end

%% Feed-forward (model) function for the neural network
function dlU = model(parameters, dlXBC)

    dlXT = dlXBC;
    numLayers = numel(fieldnames(parameters))/2;
    
    % First fully connect operation.
    weights = parameters.fc1_Weights;
    bias = parameters.fc1_Bias;
    dlU = fullyconnect(dlXT,weights,bias);
    
    % tanh and fully connect operations for remaining layers.
    for i = 2:numLayers
        name = "fc" + i;
    
        dlU = tanh(dlU);
    
        weights = parameters.(name + "_Weights");
        bias = parameters.(name + "_Bias");
        dlU = fullyconnect(dlU, weights, bias);
    end
    
end