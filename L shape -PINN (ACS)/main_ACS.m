
clc; clear; close all;
rng(39);
tic;
warning('off', 'all'); % this helps to ignore all warnings 
n = 500;
r = 8 * pi;
w1 = r * (2 * rand(n, 2) - 1);
b1 = r * (2 * rand(n, 1) - 1);

% Collocation points in L-shape domain
m2 = 900;
[x_colloc(:,1), x_colloc(:,2), xd] = generate_L_shape_points(m2, 5);
x_vec = x_colloc;

% Feature evaluation and Laplacian components
WdotXplusB = w1 * x_colloc' + b1;       % [n x m2]
phi = cos(WdotXplusB);                  % [n x m2]

d2y_dx2 = (-phi .* (w1(:,1).^2))';      % [m2 x n]
d2y_dy2 = (-phi .* (w1(:,2).^2))';      % [m2 x n]
w_x     = phi';                         % [m2 x n] n

% Final operator matrices
A = -(d2y_dx2 + d2y_dy2);  % Laplacian
B = w_x;

% Boundary points (180 points on L-shape)
bc1 = [linspace(-1, 1, 60)', -1 * ones(60,1)];
bc2 = [1 * ones(60,1), linspace(-1, 1, 60)'];
bc3 = [linspace(1, 0, 30)', 1 * ones(30,1)];
bc4 = [0 * ones(30,1), linspace(1, 0, 30)'];
bc5 = [linspace(0, -1, 30)', 0 * ones(30,1)];
bc6 = [-1 * ones(30,1), linspace(0, -1, 30)'];
boundary_points = [bc1; bc2; bc3; bc4; bc5; bc6];

x_bc = cos(w1 * boundary_points' + b1)';    % [180 x n]
y_bc = zeros(size(x_bc,1), 1);

x_amp = cos(w1 * xd' + b1)';
y_amp = ones(size(x_amp,1), 1);

%% Parameters
MaxIter = 150;
generations = 4;
guesses = 4;                   % population size per generation

init_guess_eigen = 50 * rand(1, guesses);  % Initial guesses array

final_lambdas = [];
final_weights = [];
prevPop = [];

%%  Biconvex optimisation -population based 
for gen = 1:generations
    fprintf("Generation %d\n", gen);

    [lambda_arr, weight_arr] = getEigenValWeights(init_guess_eigen, x_vec, prevPop, ...
         A, B, w1, b1, x_bc, y_bc, x_amp, y_amp, MaxIter, m2);

    for i = 1:length(lambda_arr)
        lambda = lambda_arr(i);
        weight = weight_arr(:, i);

        % Check uniqueness using eigenvalue proximity
        is_unique = true;
        for k = 1:length(final_lambdas)
            if abs(final_lambdas(k) - lambda) < 0.5
                is_unique = false;
                break;
            end
        end

        if is_unique
            final_lambdas(end+1) = lambda;
            final_weights(:, end+1) = weight;
            prevPop = final_weights;
            fprintf("  Added eigenvalue: %.6f\n", lambda);
        else
            fprintf("  Skipped near-duplicate eigenvalue: %.6f\n", lambda);
        end
    end

    init_guess_eigen = 50 * rand(1, guesses);  % Initial guesses array

end

% Sort eigenvalues and weights
[final_lambdas, sortIdx] = sort(final_lambdas);
final_weights = final_weights(:, sortIdx);

fprintf("\nFinal eigenvalues (sorted):\n");
disp(final_lambdas);
fprintf("Elapsed time: %.4f seconds\n", toc);

fd_modes =  getFd_modes(final_weights, w1, b1, final_lambdas);


%  funciton computes the eigen modes through finite difference method 
function fd_modes = getFd_modes(final_weights, w1, b1, final_lambdas)

    % Parameters
    L = 1;              % Size of square domain [-L, L] x [-L, L]
    h = 0.01;            % Grid spacing
    n = 2*L/h + 1;      % Number of grid points (odd number)
    x = linspace(-L, L, n);
    y = linspace(-L, L, n);
    [X, Y] = meshgrid(x, y);

    % Define L-shaped domain: remove top-right square [0,1]x[0,1]
    domain = ones(n, n);                     
    domain((X < 0) & (Y > 0)) = 0;         

    % Indexing unknowns
    idx = zeros(n, n);  
    count = 0;
    for i = 2:n-1
        for j = 2:n-1
            if domain(i,j) == 1
                count = count + 1;
                idx(i,j) = count;
            end
        end
    end

    N = count;                
    A = sparse(N, N);         

    % Build Laplace matrix using 5-point stencil
    for i = 2:n-1
        for j = 2:n-1
            if domain(i,j) == 1
                row = idx(i,j);
                A(row, row) = 4;

                if domain(i-1,j) == 1 && idx(i-1,j) > 0
                    A(row, idx(i-1,j)) = -1;
                end
                if domain(i+1,j) == 1 && idx(i+1,j) > 0
                    A(row, idx(i+1,j)) = -1;
                end
                if domain(i,j-1) == 1 && idx(i,j-1) > 0
                    A(row, idx(i,j-1)) = -1;
                end
                if domain(i,j+1) == 1 && idx(i,j+1) > 0
                    A(row, idx(i,j+1)) = -1;
                end
            end
        end
    end

   
    numModes = length(final_lambdas);  % Number of eigenvalues/modes to compute
    [L_evecs, lambda_vals] = eigs(A / h^2, numModes, 'smallestabs');

    % Sort eigenvalues
    [lambda_sorted, order] = sort(diag(lambda_vals));
    L_evecs = L_evecs(:, order);
    % lambda_sorted contains your fd_lambdas to use elsewhere
    fd_lambdas = lambda_sorted;
    for mode = 1:numModes
        eigvec = L_evecs(:, mode);
        norm_val = norm(eigvec);
        fprintf('Norm of eigenvector %d: %f\n', mode, norm_val);
    end
    % === Reconstruct eigenmodes on full 2D grid ===
    fd_modes = cell(1, numModes);

    for mode = 1:numModes
        eigvec = L_evecs(:, mode);
        eigvec = eigvec / norm(eigvec); % normalize for fair comparison

        Z_fd = nan(n, n);  % allocate grid with NaN for removed region
        for i = 2:n-1
            for j = 2:n-1
                if domain(i,j) == 1
                    row = idx(i,j);
                    Z_fd(i,j) = eigvec(row);
                end
            end
        end
        
        fd_modes{mode} = Z_fd; % store this mode
    end

    fprintf("finite difference eigenvalues (sorted):\n");
    disp(fd_lambdas);
    
    plotErrorModes_onFDGrid_allSaved(final_lambdas, final_weights, w1, b1, fd_modes, X, Y, domain);
end

%% Function: Generate L-shaped domain points
function [X, Y, xd] = generate_L_shape_points(N, maxi)
    X = zeros(N, 1);
    Y = zeros(N, 1);
    count = 0;
    while count < N
        x = 2 * rand - 1;
        y = 2 * rand - 1;
        if ~(x < 0 && y > 0) && x ~= 0 && y ~= 0
            count = count + 1;
            X(count) = x;
            Y(count) = y;
        end
    end

    xd = zeros(maxi, 2);
    count = 0;
    while count < maxi
        x = 2 * rand - 1;
        y = 2 * rand - 1;
        if ~(x < 0 && y > 0) && x ~= 0 && y ~= 0
            count = count + 1;
            xd(count, :) = [x, y];
        end
    end
end

