clc; clear; close all;
warning('off', 'all'); % this helps to ignore all warnings
rng(39);

tic;

% Random feature setup
n = 500;               
r = pi;   % bandwidth               
w1 = r*(2*rand(n,2)-1);  
b1 = r*(2*rand(n,1)-1);  

% Collocation points generation
m2 = 1500; 
a = 10; b = 5;
%  collocation points 
[x_colloc(:,1), x_colloc(:,2), xd] = generate_Rect_shape_points(m2, a, b, 5); % 8
%  domain for monte carlo integral 
x_vec = zeros(m2,2);

[x_vec(:,1) ,x_vec(:,2),~]=generate_Rect_shape_points(m2, a ,b ,0);

%  amplitute constraint
x_amp = cos(w1 * xd' + b1)';
y_amp = ones(size(x_amp, 1), 1);

% Residual construction
Phi_ = cos(w1 * x_colloc' + b1);

d4w_dx4 =  (Phi_ .* (w1(:,1).^4))'; 
d4w_dy4 =  (Phi_ .* (w1(:,2).^4))'; 
d4w_dx2dy2 = (Phi_ .*(w1(:,1).^2).* (w1(:,2).^2))' ;
w_x     =  Phi_'; 
YPDE = zeros(m2, 1);

% Boundary condition constraint
bc1 = [linspace(0, a, 120)',  zeros(120,1)]; %x,0
bc2 = [a * ones(60,1), linspace(0, b, 60)'];  % a,y
bc3 = [linspace(0,a, 120)', b * ones(120,1)]; % x,b
bc4 = [zeros(60,1), linspace(0, b, 60)']; % 0,y

boundary_points = [bc1; bc2; bc3; bc4];

nu = 0.3; % poisson ratio  

% 1.  dirchlet BC deflection is zero 
x_bc1 = cos(w1 * boundary_points' + b1)';
% 2. moment is zero 
hor_edges = [bc1; bc3];
ver_edges =[bc2;bc4];


M11 = ((-cos(w1 * ver_edges' + b1) .* (w1(:,1).^2))') + nu*((-cos(w1 * ver_edges' + b1) .* (w1(:,2).^2))');

M22 = ((-cos(w1 * hor_edges' + b1) .* (w1(:,2).^2))') + nu*((-cos(w1 * hor_edges' + b1) .* (w1(:,1).^2))');


x_bc2 =[M11;M22];

x_bc = [x_bc1 ;x_bc2];
y_bc = zeros(size(x_bc,1),1);

%  governing equation (LHs and rhs )
A = (d4w_dy4 +2*d4w_dx2dy2+ d4w_dx4);
B = w_x;

analyticalEigenVals= [0.243,0.623,1.646,2.815,3.896,6.088,8.192,9.974,13.335,15.585];
% Parameters
MaxIter = 25;

generations = 4;
guesses = 5;                   % population size per generation
scaling_fac =15;
init_guess_eigen = (scaling_fac * rand(1, guesses));  % Initial guesses array

final_lambdas = [];
final_weights = [];
prevPop = [];
eigen_convergence = {};   % cell array instead of []
loss_convergence  = {};   % cell array instead of []
for gen = 1:generations
    fprintf("Generation %d\n", gen);

    [lambda_arr, weight_arr, currGenLossConv, CurrGenEigenValConv] = getEigenValWt( init_guess_eigen, x_vec, prevPop, ...
        A, B, w1, b1, x_bc, y_bc, x_amp, y_amp, MaxIter, m2 , n);

    for i = 1:length(lambda_arr)
        lambda = lambda_arr(i);
        weight = weight_arr(:, i);

        % Check uniqueness using eigenvalue proximity
        is_unique = true;
        for k = 1:length(final_lambdas)
            if abs(final_lambdas(k) - lambda) < 0.3
                is_unique = false;
                break;
            end
        end

        if is_unique
            % Store convergence histories as variable-length column vectors
            eigen_convergence{end+1} = CurrGenEigenValConv(:, i);
            loss_convergence{end+1}  = currGenLossConv(:, i);

            % Store eigenvalue and weight
            final_lambdas(end+1) = lambda;
            final_weights(:, end+1) = weight;
            prevPop = final_weights;

            fprintf("  Added eigenvalue: %.6f\n", lambda);
        else
            fprintf("  Skipped near-duplicate eigenvalue: %.6f\n", lambda);
        end
    end

    init_guess_eigen = scaling_fac * rand(1, guesses);  % Initial guesses array

end

% Sort eigenvalues and weights
[final_lambdas, sortIdx] = sort(final_lambdas);
final_weights = final_weights(:, sortIdx);

% Reorder convergence histories according to sorted eigenvalues
loss_convergence  = loss_convergence(sortIdx);
eigen_convergence = eigen_convergence(sortIdx);

fprintf("\nFinal eigenvalues (sorted):\n");
disp(final_lambdas);
fprintf("Elapsed time: %.4f seconds\n", toc);

plot_(final_lambdas, final_weights, w1, b1, loss_convergence, eigen_convergence,analyticalEigenVals, a, b, 20);


function [X, Y, xd] = generate_Rect_shape_points(N, l, b, maxi)
    X = zeros(N,1); Y = zeros(N,1);
    for i = 1:N
        X(i) = (l) * rand;
        Y(i) = (b) * rand;
    end

    xd = zeros(maxi, 2);
    for i = 1:maxi
        xd(i,1) = (l ) * rand;
        xd(i,2) = (b ) * rand;
    end
    
    % plotting the domain (sampled points)
    % figure;
    % scatter(X, Y, 10, 'filled');
    % title('Collocation Points');
    % xlabel('x'); ylabel('y'); axis equal;

    % figure;
    % scatter(xd(:,1), xd(:,2), 10, 'filled');
    % title('Anchor Points');
    % xlabel('x'); ylabel('y'); axis equal;

end