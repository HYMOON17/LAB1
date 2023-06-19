%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear; 
close all;

%% first extract the regular-dose images in each patient foler %%%%%%%%
ImagePath = '~/Desktop/data/MayoData/L096/full_3mm/';

s = dir([ImagePath, '*.IMA']);
allNames = {s.name};
fileNum = max(size(allNames));
for k = 1:1:fileNum
    str = [char(ImagePath),char(allNames(1,k))]; 
    xfdk(:,:,k) = dicomread(str);
end
xfdk = single(xfdk);
figure; im('mid3',permute(xfdk,[2 1 3]),[800 1200]);colormap(gca,gray);drawnow
% unit of readout fdk images: (modified-)HU
save('~/Desktop/data/MayoData/L096/full_3mm.mat', 'xfdk');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% training data
load('~/Desktop/data/MayoData/L096/full_3mm.mat');  % unit of the loaded images: HU
xfdk1 = xfdk(:,:,40:114);
% figure; im('mid3',permute(xfdk,[2 1 3]),[800 1200]);colormap(gca,gray);drawnow
% figure; im('mid3',permute(xfdk1,[2 1 3]),[800 1200]);colormap(gca,gray);drawnow
% figure;imshow(xfdk1(:,:,1),[800 1200])
% figure;imshow(xfdk1(:,:,end),[800 1200])
% 
load('~/Desktop/data/MayoData/L143/full_3mm.mat');  % unit of the loaded images: HU
xfdk2 = xfdk(:,:,30:104);
% figure; im('mid3',permute(xfdk,[2 1 3]),[800 1200]);colormap(gca,gray);drawnow
% figure; im('mid3',permute(xfdk2,[2 1 3]),[800 1200]);colormap(gca,gray);drawnow 
% figure;imshow(xfdk2(:,:,1),[800 1200])
% figure;imshow(xfdk2(:,:,end),[800 1200])
% 
load('~/Desktop/data/MayoData/L333/full_3mm.mat');  % unit of the loaded images: HU
xfdk3 = xfdk(:,:,1:20);
% figure; im('mid3',permute(xfdk,[2 1 3]),[800 1200]);colormap(gca,gray);drawnow
% figure; im('mid3',permute(xfdk3,[2 1 3]),[800 1200]);colormap(gca,gray);drawnow
% figure;imshow(xfdk3(:,:,1),[800 1200])
% figure;imshow(xfdk3(:,:,end),[800 1200])
% 
xtrue = cat(3,xfdk1,xfdk2, xfdk3);
% load('./train_imgs.mat');  

%% test data
% load('~/Desktop/data/MayoData/L291/full_3mm.mat');  % unit of the loaded images: HU
% xtrue = xfdk(:,:,51:80);
% figure; im('mid3',permute(xfdk,[2 1 3]),[800 1200]);colormap(gca,gray);drawnow
% figure; im('mid3',permute(xtrue,[2 1 3]),[800 1200]);colormap(gca,gray);drawnow
% figure;imshow(xtrue(:,:,1),[800 1200])
% figure;imshow(xtrue(:,:,end),[800 1200])
% load('./test_imgs.mat');

%% setup system and image geometry
I0 = 5e4;

sg = sino_geom('fan', 'units', 'mm', 'nb',736, 'na',1152,'orbit',360, 'ds',1.2858,...
     'strip_width','d','dsd',1085.6,'dso',595,'dfs',0);

ig = image_geom('nx',512,'fov', sg.rfov*sqrt(2)); % dx=0.69298mm, fov=354.8065mm

A = Gtomo2_dscmex(sg, ig, 'nthread', jf('ncore'));

mm2HU = 1000 / 0.02;


num = size(xtrue,3);
%% fbp recon
for ii = 1:num
    
    fprintf('Slice #%d of %d:\n', ii, num);
    figure(44); imshow(xtrue(:,:,ii), [800 1200]);drawnow
    
    sino_true = A * xtrue(:,:,ii);
 
    % adding poisson noise
    yi = poisson(I0 * exp(-sino_true ./mm2HU), 0, 'factor', 0.4);    
    sino = log(I0./yi) * mm2HU;
       
    tmp = fbp2(sg, ig);
    xfbp1 = fbp2(sino, tmp, 'window', ''); clear tmp; % ramp filter
    xfbp1 = max(xfbp1, 0);
    
    figure(55); imshow(xfbp1, [800 1200]);drawnow
    
    xfbp(:,:,ii) = xfbp1;

end


save('./data/train_imgs.mat', 'xtrue', 'xfbp');
% save('./data/test_imgs.mat', 'xtrue', 'xfbp');







