function If = nonLocalMeans_Cuda(I, patchSize, filtSigma, patchSigma)
    % CUDA Implementation of non Local Means algorithm

    % 1024 Threads / Block arranged in a square
    threadsPerBlock = [16 16];

    m = size(I,1);
    n = size(I,2);

    % Mirror Image to add padding for algorithm boundary conditions
    mirrored_I = [flipud(fliplr(I)) flipud(I) fliplr(flipud(I));...
        fliplr(I) I fliplr(I);fliplr(flipud(I)) flipud(I) flipud(fliplr(I))];

    patchRadius_x = (patchSize(1) - 1) / 2;
    patchRadius_y = (patchSize(2) - 1) / 2 ;

    % Pad mirrored pixels to image (depends on patchSize)
    mirrored_I = mirrored_I((m+1)-patchRadius_x:(2*m)+patchRadius_x, ...
        (n+1)-patchRadius_y:(2*n)+patchRadius_y);

    %% Kernel Setup

    kernel = parallel.gpu.CUDAKernel( '../cuda/non_local_means_kernel.ptx', ...
                                   '../cuda/non_local_means_kernel.cu');

    numBlocks = ceil ( [m n] ./ threadsPerBlock );
    kernel.ThreadBlockSize = threadsPerBlock;
    kernel.GridSize = numBlocks;

    % Shared memory allocation for gaussian filter +
    % the two pixel patches we are going to compare
    kernel.SharedMemorySize = patchSize(1)*patchSize(2)*4 ...
        + 2*(threadsPerBlock(1)+2*patchRadius_x)*(threadsPerBlock(2)+2*patchRadius_y)*4;

    % Device matrices
    d_filtered_I = gpuArray(zeros(m,n, 'single'));
    d_I = gpuArray(single(mirrored_I));
    d_patchSize = gpuArray(int32(patchSize));

    disp('Kernel Execution Time');
    tic;
    d_filtered_I = feval(kernel, d_I, d_filtered_I, m, n,...
        single(patchSigma), d_patchSize, single(filtSigma));
    wait(gpuDevice);
    toc;

    disp('Data gather time');
    tic;
    If = gather(d_filtered_I);
    toc;

    If = reshape(If, [m n]);

end
