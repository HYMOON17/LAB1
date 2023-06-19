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

load('~/Desktop/data/MayoData/L143/full_3mm.mat');  % unit of the loaded images: HU
xfdk2 = xfdk(:,:,30:104);
% figure; im('mid3',permute(xfdk,[2 1 3]),[800 1200]);colormap(gca,gray);drawnow
% figure; im('mid3',permute(xfdk2,[2 1 3]),[800 1200]);colormap(gca,gray);drawnow 
% figure;imshow(xfdk2(:,:,1),[800 1200])
% figure;imshow(xfdk2(:,:,end),[800 1200])

load('~/Desktop/data/MayoData/L333/full_3mm.mat');  % unit of the loaded images: HU
xfdk3 = xfdk(:,:,1:20);
% figure; im('mid3',permute(xfdk,[2 1 3]),[800 1200]);colormap(gca,gray);drawnow
% figure; im('mid3',permute(xfdk3,[2 1 3]),[800 1200]);colormap(gca,gray);drawnow
% figure;imshow(xfdk3(:,:,1),[800 1200])
% figure;imshow(xfdk3(:,:,end),[800 1200])

xtrue = cat(3,xfdk1,xfdk2, xfdk3);
% load('./train_imgs.mat'); 

%% test data
% load('./Desktop/data/MayoData/L291/full_3mm.mat');  % unit of the loaded images: HU
% xtrue = xfdk(:,:,51:80);
% figure; im('mid3',permute(xfdk,[2 1 3]),[800 1200]);colormap(gca,gray);drawnow
% figure; im('mid3',permute(xtrue,[2 1 3]),[800 1200]);colormap(gca,gray);drawnow
% figure;imshow(xtrue(:,:,1),[800 1200])
% figure;imshow(xtrue(:,:,end),[800 1200])
% % load('./test_imgs.mat'); 

%% setup system and image geometry
I0 = 5e4;
down = 2;

sg = sino_geom('fan', 'units', 'mm', 'nb',736, 'na',1152,'orbit',360, 'ds',1.2858,...
     'strip_width','d','dsd',1085.6,'dso',595,'dfs',0);
 
sg_odd = sino_geom('fan', 'units', 'mm', 'nb',736, 'na',1152/down,'orbit',360, 'ds',1.2858,...
     'strip_width','d','dsd',1085.6,'dso',595,'dfs',0, 'orbit_start',0);
 
sg_even = sino_geom('fan', 'units', 'mm', 'nb',736, 'na',1152/down,'orbit',360, 'ds',1.2858,...
     'strip_width','d','dsd',1085.6,'dso',595,'dfs',0, 'orbit_start',360/1152);

ig = image_geom('nx',512,'fov', sg.rfov*sqrt(2)); % dx=0.69298mm, fov=354.8065mm

A = Gtomo2_dscmex(sg, ig, 'nthread', jf('ncore'));

fbp_recon_full = fbp2(sg, ig);
fbp_recon_odd = fbp2(sg_odd, ig);
fbp_recon_even = fbp2(sg_even, ig);

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
    
    % downsample
    sino_old = sino(:,1:down:end-1);   
    sino_even = sino(:,2:down:end);   

    xfbp_full = fbp2(sino, fbp_recon_odd, 'window', ''); % ramp filter    
    xfbp1 = fbp2(sino_old, fbp_recon_odd, 'window', ''); 
    xfbp2 = fbp2(sino_even, fbp_recon_even, 'window', ''); 
    xfbp_full = max(xfbp_full, 0);       
    xfbp1 = max(xfbp1, 0);   
    xfbp2 = max(xfbp2, 0);    

%     figure(77);im_toggle(xref',xfbp1',xfbp2',[800 1200]);colormap(gray),drawnow
    xfbp(:,:,ii) = xfbp_full;
    xfbp_odd(:,:,ii) = xfbp1;
    xfbp_even(:,:,ii) = xfbp2;

end


save('./train_imgs_odd_even.mat', 'xtrue', 'xfbp', 'xfbp_odd', 'xfbp_even');
% save('./test_imgs_odd_even.mat', 'xtrue', 'xfbp', 'xfbp_odd', 'xfbp_even');




