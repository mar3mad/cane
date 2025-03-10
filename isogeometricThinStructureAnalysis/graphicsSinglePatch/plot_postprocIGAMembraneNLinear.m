function index = plot_postprocIGAMembraneNLinear...
    (BSplinePatch, dHat, graph, outMsg)
%% Licensing
%
% License:         BSD License
%                  cane Multiphysics default license: cane/license.txt
%
% Main authors:    Andreas Apostolatos
%
%% Function documentation
%
% Plots two windows in one: The first contains the reference and/or the
% current configuration of an isogeometric membrane given the Control Point 
% displacement. The second window contains the visualization of the 
% selected resultant component over the plate's domain in the reference 
% configuration for the nonlinear analysis.
%
%                    Input :
%           BSplinePatches : Its an array of structures {patch1,patch2,...}
%                            each of the patch structures containing the
%                            following information
%                                 .p,.q: Polynomial degrees
%                              .Xi,.Eta: knot vectors
%                                   .CP: Control Points coordinates and 
%                                        weights
%                              .isNURBS: Flag on whether the basis is a 
%                                        NURBS or a B-Spline
%                             .homDOFs : The global numbering of the
%                                        DOFs where homogeneous Dirichlet
%                                        boundary conditions are applied
%                           .inhomDOFs : The global numbering of the
%                                        DOFs where homogeneous Dirichlet
%                                        boundary conditions are applied
%                     .valuesInhomDOFs : Prescribed values to the DOFs 
%                                        where homogeneous Dirichlet
%                                        boundary conditions are applied
%                               FGamma : The boundary applied force vector
%                                        over the B-Spline patch
%                           bodyForces : Function handle to the computation 
%                                        of the body forces
%                        .DOFNumbering : Numbering of the DOFs sorted into
%                                        a 3D array
%                          .parameters : material parameters of the shell
%                                 .int : On the numerical integration
%                                         .type : 'default' or 'user'
%                                        .xiNGP : No. of GPs along xi-
%                                                 direction for stiffness 
%                                                 entries
%                                       .etaNGP : No. of GPs along eta-
%                                                 direction for stiffness 
%                                                 entries
%                                 .xiNGPForLoad : No. of GPs along xi-
%                                                 direction for load 
%                                                 entries
%                                .etaNGPForLoad : No. of GPs along eta-
%                                                 direction for load 
%                                                 entries
%                                   .nGPForLoad : No. of GPs along boundary
%                               dHat : The displacement field
%                              graph : Information on the graphics
%                             outMsg : Whether or not to output message on 
%                                      refinement progress
%                                          'outputEnabled' : enables output 
%                                                            information
%
%                             Output :
%                              index : The index of the current graph
%               
%                                      graphics
%
% Function layout :
%
% 0. Read input
%
% 1. Plot the first window: Reference and/or current configuration
%
% 2. Plot the second window: Resultant visualization
%
% 3. Update the graph index
%
% 4. Appendix
%
%% Function main body
if strcmp(outMsg,'outputEnabled')
    fprintf('_________________________________________________________________________\n');
    fprintf('#########################################################################\n');
    fprintf('Plotting postprocessing configuration and resultant visualization for the\n');
    fprintf('nonlinear isogeometric Kirchhoff-Love shell has been initiated\n\n');
    fprintf('Configuration to be visualized (1st window): ');
    if strcmp(graph.postprocConfig,'reference')
        fprintf('Reference\n');
    elseif strcmp(graph.postprocConfig,'current')
        fprintf('Current\n');
    elseif strcmp(graph.postprocConfig,'referenceCurrent')
        fprintf('Reference and current\n');
    end
    fprintf('Resultant to be visualized (2nd window): ');
    if strcmp(graph.resultant,'displacement')
        if strcmp(graph.component,'x')
            fprintf('Displacement component u_x\n');
        elseif strcmp(graph.component,'y')
            fprintf('Displacement component u_y\n');
        elseif strcmp(graph.component,'z')
            fprintf('Displacement component u_z\n');
        elseif strcmp(graph.component,'2norm')
            fprintf('Displacement magnitude ||u||_2\n');
        end
    elseif strcmp(graph.resultant,'strain')
        if strcmp(graph.component,'1')
            fprintf('Strain component epsilon_11\n');
        elseif strcmp(graph.component,'2')
            fprintf('Strain component epsilon_22\n');
        elseif strcmp(graph.component,'12')
            fprintf('Strain component epsilon_12\n');
        elseif strcmp(graph.component,'1Principal')
            fprintf('1st pricipal strain field \epsilon_1\n');
        elseif strcmp(graph.component,'2Principal')
            fprintf('2nd pricipal strain field \epsilon_2\n');
        end
    elseif strcmp(graph.resultant,'force')
        if strcmp(graph.component,'1')
            fprintf('Force component n_11\n');
        elseif strcmp(graph.component,'2')
            fprintf('Force component n_22\n');
        elseif strcmp(graph.component,'12')
            fprintf('Force component n_12\n');
        elseif strcmp(graph.component,'1Principal')
            fprintf('1st pricipal force field n_1\n');
        elseif strcmp(graph.component,'2Principal')
            fprintf('2nd pricipal force field n_2\n');
        end
    else
        error('No resultant has been chosen to be visualized\n');
    end
    fprintf('_________________________________________________________________________\n\n');

    % start measuring computational time
    tic;
end

%% 0. Read input

% Retrieve the patch properties
p = BSplinePatch.p;
q = BSplinePatch.q;
Xi = BSplinePatch.Xi;
Eta = BSplinePatch.Eta;
CP = BSplinePatch.CP;
isNURBS = BSplinePatch.isNURBS;
homDOFs = BSplinePatch.homDOFs;
parameters = BSplinePatch.parameters;
DOFNumbering = BSplinePatch.DOFNumbering;
if exist('BSplinePatch.FGamma','var')
    FGamma = BSplinePatch.FGamma;
else
    FGamma = zeros(3*length(CP(:,1,1))*length(CP(1,:,1)),1);
end

% Initialize handle to the figure
figure(graph.index)

% Grid point number for the plotting of both the B-Spline surface the knots
% as well as the resultant computation over the domain
xiGrid = 49;
etaGrid = 49;

%% 1. Plot the first window: Reference and/or current configuration

% Plot the window
subplot(2,1,1);
plot_postprocCurrentConfigurationIGAThinStructure...
    (p,q,Xi,Eta,CP,isNURBS,xiGrid,etaGrid,homDOFs,FGamma,dHat,graph);

% Assign graphic properties and title
camlight left; lighting phong;
axis equal;
xlabel('x','FontSize',14);
ylabel('y','FontSize',14);
if strcmp(graph.postprocConfig,'reference')
    title('Reference configuration');
elseif strcmp(graph.postprocConfig,'current')
    title('Current configuration/nonlinear');
elseif strcmp(graph.postprocConfig,'referenceCurrent')
    title('Reference and current configuration/nonlinear');
end
hold off;

%% 2. Plot tNBC = {NBC};he second window: Resultant visualization

% Plot the window
subplot(2,1,2);
plot_postprocResultantsIGAMembraneNLinear...
    (p,q,Xi,Eta,CP,isNURBS,parameters,DOFNumbering,xiGrid,etaGrid,dHat,graph);

% Assign graphic properties and title
if strcmp(graph.resultant,'displacement')
    if strcmp(graph.component,'x')
        titleString = 'Displacement component d_x';
    elseif strcmp(graph.component,'y')
        titleString = 'Displacement component d_y';
    elseif strcmp(graph.component,'z')
        titleString = 'Displacement component d_z';
    elseif strcmp(graph.component,'2norm')
        titleString = 'Displacement magnitude ||d||_2';
    end
elseif strcmp(graph.resultant,'strain')
    if strcmp(graph.component,'1')
        titleString = 'Strain component \epsilon_{11}';
    elseif strcmp(graph.component,'2')
        titleString = 'Strain component \epsilon_{22}';
    elseif strcmp(graph.component,'12')
        titleString = 'Strain component \epsilon_{12}';
    elseif strcmp(graph.component,'1Principal')
        titleString = '1st pricipal strain field \epsilon_1';
    elseif strcmp(graph.component,'2Principal')
        titleString = '2nd pricipal strain field \epsilon_2';
    end
elseif strcmp(graph.resultant,'force')
    if strcmp(graph.component,'1')
        titleString = 'Force component n^{11}';
    elseif strcmp(graph.component,'2')
        titleString = 'Force component n^{22}';
    elseif strcmp(graph.component,'12')
        titleString = 'Force component n^{12}';
    elseif strcmp(graph.component,'1Principal')
        titleString = '1st pricipal force field n^1';
    elseif strcmp(graph.component,'2Principal')
        titleString = '2nd pricipal force field n^2';
    end
end
title(titleString);

shading interp;
colormap('jet');

% invert default colormap => red = negativ, blue = positive
% COL = colormap;
% invCOL(:,1) = COL(:,3);
% invCOL(:,2) = COL(:,2);
% invCOL(:,3) = COL(:,1);
% colormap(invCOL);

% make colormap symmetric
% colim = caxis;
% caxis([-max(abs(colim)) max(abs(colim))]);
colorbar;
axis equal;
xlabel('x','FontSize',14);
ylabel('y','FontSize',14);

%% 3. Update the graph index
index = graph.index + 1;

%% 4. Appendix
if strcmp(outMsg,'outputEnabled')
    % Save computational time
    computationalTime = toc;

    fprintf('Plotting the current configuration took %.2d seconds \n\n',computationalTime);
    fprintf('__________________Plotting Current Configuration Ended___________________\n');
    fprintf('#########################################################################\n\n\n');
end

end
