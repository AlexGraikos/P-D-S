function pipeline_non_local_means(image_name, patch_size)

  %% PARAMETERS
  
  % input image
  pathImg = strcat('../data/', image_name, '.mat');
  strImgVar = image_name;
  
  % noise
  noiseParams = {'gaussian', ...
                 0,...
                 0.001};
  
  % filter sigma value
  filtSigma = 0.02;
  patchSize = patch_size;
  patchSigma = 5/3;
  
  %% USEFUL FUNCTIONS

  % image normalizer
  normImg = @(I) (I - min(I(:))) ./ max(I(:) - min(I(:)));
  
  %% (BEGIN)

  fprintf('...begin %s...\n',mfilename);  
  
  %% INPUT DATA
  
  fprintf('...loading input data...\n')
  
  ioImg = matfile( pathImg );
  I     = ioImg.(strImgVar);
  
  %% PREPROCESS
  
  fprintf(' - normalizing image...\n')
  I = normImg( I );
  
  figure('Name','Original Image');
  imagesc(I); axis image;
  colormap gray;
  savefig(strcat('../results/',image_name,'-',int2str(patchSize),'-',' Original Image'));
  
  %% NOISE
  
  fprintf(' - applying noise...\n')
  J = imnoise( I, noiseParams{:} );
  figure('Name','Noisy-Input Image');
  imagesc(J); axis image;
  colormap gray;
  savefig(strcat('../results/',image_name,'-',int2str(patchSize),'-',' Noisy-Input Image'));
  
  %% NON LOCAL MEANS
  
  disp('Executing NLM Kernel');
  A = nonLocalMeans_Cuda(J, patchSize, filtSigma, patchSigma);
  A = double(A);
  
  disp('Executing Matlab script');
  If = nonLocalMeans( J, patchSize, filtSigma, patchSigma );

  disp('CUDA Error');
  disp(immse(I,A));
  disp('Original error');
  disp(immse(I,If));
  disp('Difference');
  disp(immse(A,If));
  
  
  %% VISUALIZE RESULT
  
  figure('Name', 'CUDA Filtered image');
  imagesc(A); axis image;
  colormap gray;
  savefig(strcat('../results/',image_name,'-',int2str(patchSize),'-',' CUDA Filtered Image'));
  
  figure('Name', 'Filtered image');
  imagesc(If); axis image;
  colormap gray;
  savefig(strcat('../results/',image_name,'-',int2str(patchSize),'-',' Matlab Filtered Image'));
  
  figure('Name', 'Residual');
  imagesc(A-J); axis image;
  colormap gray;
  savefig(strcat('../results/',image_name,'-',int2str(patchSize),'-',' Residual'));
  
  %% (END)

  fprintf('...end %s...\n',mfilename);
  
end
