clc
clear all
close all

PERSON = 2;
loop_mode = 1; % if loop mode is on, uncommented below codes; if not, commented below codes. (Also for the codes at the end)


% Global variables
effective_range_bins_array = zeros(60,120);
effective_range_bins = zeros(1,120);
in_situ_feature = zeros(25,120);


for person_number = 1:PERSON
    for motion_label = 1:12
        for data_number = 1:10
motion_label
data_number


% motion_label = 7
% data_number = 2

%% Parameter setting

% Function switch
avg_true = 0;
median_true = 1; 
mti_mode = 3; % 2: MTI2; 3: MTI3
binarization_mode = 2; % 1: ostu thresholding; 2: manually set thresholding

% Plots switch
plot_switch = 0;

raw_data_plot = 1;
selected_data_plot = 0;
avg_data_plot = 0;
mti2_data_plot = 0;
mti3_data_plot = 0;
median_data_plot = 1;
otsu_plot = 0;
stft_plot = 0;

% Parameters
AMF_THRESHOLD = 10; % Adaptive Mean Filter Threshold, choose from 0-25
ISOLATED_THRESHOLD = 7;
isolated_r = 7; % isolated searching radius
SITU_THRESHOLD = 21; 
WINDOW_LENGTH = 32;
% ALPHA = 0.15;
TORSO_ENERGY = 0.004;
OSTU_LEVEL = 32;
MANUAL_THRESHOLD = 0.0015;

% Read & write mat files
write_mode_on = 0; % 0: read data from edited raw_data.mat file with range from select.mat;
                   % 1: write each .dat raw file into raw_data.mat files and select range stored in select.mat;
                   % 2: write only selected range into select.mat;
                   % 3: read raw data from .dat only without any editing to .mat files;

% person_number = 1;
% motion_label = 12; % If write mode is on: select the data to write into .mat file; else read data from .mat file.
% data_number = 9;


% Data selection
origin_width = 600;
fast_time_height = 60;
width = 400;

% start = 90;
% finish = start + width - 1;



if write_mode_on == 1 || write_mode_on == 3
    %% Original XeThru Codes %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    file ='.\xethru_datafloat_20200618_175903.dat'; 
    % file ='.\xethru_datafloat_20200603_135925.dat'; % a standing still
    % file ='.\xethru_datafloat_20200603_135951.dat'; % b bowing
    % file ='.\xethru_datafloat_20200603_140032.dat'; % c sit/stand xethru_datafloat_20200605_104708
    % file ='.\xethru_datafloat_20200603_140115.dat'; % d squating
    % file ='.\xethru_datafloat_20200603_140157.dat'; % e jumping
    % file ='.\xethru_datafloat_20200603_140317.dat'; % f falling vertically

    % file ='.\xethru_datafloat_20200603_140459.dat'; % g walking
    % file ='.\xethru_datafloat_20200603_140538.dat'; % h elderly

    % % file ='.\xethru_datafloat_20200603_140620.dat'; % i jogging % Problem
    % % file ='.\xethru_datafloat_20200603_140644.dat'; % j jumping forward % Problem

    % file ='.\xethru_datafloat_20200603_140711.dat'; % k falling forward
    % file ='.\xethru_datafloat_20200603_140752dat'; % l crawling forward


    NumHdrs = 7;
    Num32bitEpoch = 2;

    fid = fopen(file, 'rb');
    if fid < 3
        disp(['couldnt read file ' file]);
        return
    end

    f = dir(file);
    fsize = f.bytes;

    % read first frame
    ctr = 0;
    hdrMat = [];
    FrameMat = [];
    TimeVec = [];

    while (1)
        if feof(fid)
            break
        end

        % read header
        contentID = fread(fid, 1,'uint32');
        frameCtr = fread(fid, 1,'uint32');
        data_length = fread(fid, 1,'uint32');


        % check valid header read
        if  isempty(contentID) || isempty(frameCtr) || isempty(data_length) 
            break;
        end
        
        data = fread(fid,data_length, 'single');
        ctr = ctr + 1;
        i_vec(:,ctr) = data(1:data_length/2);
        q_vec(:,ctr) = data(data_length/2+1:data_length);
        iq_vec(:,ctr) = i_vec(:,ctr) + 1i*q_vec(:,ctr);
        am(:,ctr)=sqrt(i_vec(:,ctr).^2+q_vec(:,ctr).^2);

    end
end
    
%% Raw Data Vectorization and Storation %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Read
if write_mode_on ~= 3
    if person_number == 1
        mat_raw = matfile('raw_data1.mat');
        mat_label = matfile('labels1.mat');
        mat_select = matfile('select1.mat');
        mat_pred_type = matfile('pred_types1.mat');
    elseif person_number == 2
        mat_raw = matfile('raw_data2.mat');
        mat_label = matfile('labels2.mat');
        mat_select = matfile('select2.mat');
        mat_pred_type = matfile('pred_types2.mat');
    end
    
    raw_iq_data = mat_raw.raw_iq_data;
    label = mat_label.label;
    select = mat_select.select;
    pred_type = mat_pred_type.pred_type;
    same_type = zeros(1,120);
end

if write_mode_on == 0
    for i = 1:origin_width-1
        iq_vec(:,i) = raw_iq_data((i-1)*fast_time_height+1:i*fast_time_height,(motion_label-1)*10+data_number);
    end
    am = abs(iq_vec);
% Write
elseif write_mode_on == 1
    raw_iq_data(:,(motion_label-1)*10+data_number) = iq_vec(:);
    label((motion_label-1)*10+data_number) = motion_label;
    select(1,(motion_label-1)*10+data_number) = start;
    select(2,(motion_label-1)*10+data_number) = finish;
    if person_number == 1
        save('raw_data1.mat','raw_iq_data');
        save('labels1.mat','label');
        save('select1.mat','select');
    elseif person_number == 2
        save('raw_data2.mat','raw_iq_data');
        save('labels2.mat','label');
        save('select2.mat','select');
    end
    
elseif write_mode_on == 2
    for i = 1:origin_width-1
        iq_vec(:,i) = raw_iq_data((i-1)*fast_time_height+1:i*fast_time_height,(motion_label-1)*10+data_number);
    end
    am = abs(iq_vec);
    
    select(1,(motion_label-1)*10+data_number) = start;
    select(2,(motion_label-1)*10+data_number) = finish;
    if person_number == 1
        save('select1.mat','select');
    elseif person_number == 2
        save('select2.mat','select');
    end
end


%% Raw Data Visualisation %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if raw_data_plot == 1 && plot_switch == 1
    figure(1)
    surf(am) % am = amplitude
    shading interp
    view(0,90) % Top view
    axis tight
    colormap hsv
    colorbar
    title('Raw Data Visualisation');
    xlabel('Slow time/sample number)');
    ylabel('Fast time/sample number');
end

%% Data Selection %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Fs = 100; % sampling frequency
if write_mode_on == 0 % read from .mat file
    start = select(1,(motion_label-1)*10+data_number);
    finish = select(2,(motion_label-1)*10+data_number);
end
am_selected = am(:,start:finish);
iq_selected = iq_vec(:,start:finish);
% finish_time=(finish-start)/Fs;
% time=0:1/Fs:finish_time;

% Selected Data Visualisation
if selected_data_plot == 1 && plot_switch == 1
    figure(2)
    surf(am_selected)
    shading interp
    view(0,90)
    axis tight
    colormap hsv
    colorbar
    title('Selected Data Visualisation');
    xlabel('Slow time/sample number)');
    ylabel('Fast time/sample number');
end

%% Pre-processing %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Average subtraction
if avg_true == 1
    iq_avg = mean(iq_selected,2);
    iq_selected = iq_selected - iq_avg;
    am_avg = abs(iq_selected);
end

% Average Data Visualisation
if avg_data_plot == 1 && plot_switch == 1
    figure(3)
    surf(am_avg)
    shading interp
    view(0,90)
    axis tight
    colormap hsv
    colorbar
    title('Average Data Visualisation');
    xlabel('Slow time/sample number)');
    ylabel('Fast time/sample number');
end

% Moving target Indicator - Two Pulse Canceller - selected
for fast_index = 1:size(iq_selected,1)
    for slow_index = 1:size(iq_selected,2)-1
        iq_mti2(fast_index,slow_index) = iq_selected(fast_index,slow_index) - iq_selected(fast_index,slow_index+1); 
    end
end

% Moving target Indicator - Three Pulse Canceller - not selected
for fast_index = 1:size(iq_selected,1)
    for slow_index = 1:size(iq_selected,2)-2
        iq_mti3(fast_index,slow_index) = (iq_selected(fast_index,slow_index) - 2*iq_selected(fast_index,slow_index+1) + iq_selected(fast_index,slow_index+2)); 
    end
end

am_mti2 = abs(iq_mti2);
am_mti3 = abs(iq_mti3);

% MTI Data Visualisation
if mti2_data_plot == 1 && plot_switch == 1
    figure(4)
    surf(am_mti2)
    shading interp
    view(0,90)
    axis tight
    colormap hsv
    colorbar
    title('MTI2 Data Visualisation');
    xlabel('Slow time/sample number)');
    ylabel('Fast time/sample number');
end

if mti3_data_plot == 1 && plot_switch == 1
    figure(5)
    surf(am_mti3)
    shading interp
    view(0,90)
    axis tight
    colormap hsv
    colorbar
    title('MTI3 Data Visualisation');
    xlabel('Slow time/sample number)');
    ylabel('Fast time/sample number');
end

if mti_mode == 2
    am_mti = am_mti2;
elseif mti_mode == 3
    am_mti = am_mti3;
end

% Median filter
if median_true == 1
    am_median = medfilt2(am_mti,[3 3]); % a median filter of 3x3
end

if (median_data_plot == 1 && median_true == 1) && plot_switch == 1
    figure(6)
    surf(am_median)
    shading interp
    view(0,90)
    axis tight
    colormap hsv
    colorbar
    title('Median Data Visualisation');
    xlabel('Slow time/sample number)');
    ylabel('Fast time/sample number');
end

% Otsu thresholding
if binarization_mode == 1
    max_median = max(max(am_median));
    am_greylevel = am_median/max_median; % Modulate radar signal to percentage

    % figure
    [counts,x] = imhist(am_greylevel,OSTU_LEVEL);
    % stem(x,counts)
    ostu_threshold = otsuthresh(counts);

    am_bw = imbinarize(am_greylevel,ostu_threshold);

    if otsu_plot == 1 && plot_switch == 1
        figure(7)
        imshow(am_bw)
        title('Ostu Data Visualisation');
        xlabel('Slow time/sample number)');
        ylabel('Fast time/sample number');
    end
    
% Manually set thresholding
elseif binarization_mode == 2
    am_bw = imbinarize(am_median,MANUAL_THRESHOLD);
%     figure(7)
%     imshow(am_bw)
%     title('Manually set thresholding Data Visualisation');
%     xlabel('Slow time/sample number)');
%     ylabel('Fast time/sample number');
end


% Adaptive Mean filter
% https://www.csdn.net/gather_21/MtTaQg1sNDM0NDYtYmxvZwO0O0OO0O0O.html
I = am_bw;
[m,n] = size(I);
am_bw_amf = zeros(m,n);
for i = 3:m-3
    for j = 3:n-3
        sum5 = I(i-2,j-2)+I(i-2,j-1)+I(i-2,j)+I(i-2,j+1)+I(i-2,j+2)+...
               I(i-1,j-2)+I(i-1,j-1)+I(i-1,j)+I(i-1,j+1)+I(i-1,j+2)+...
               I(i,j-2)+I(i,j-1)+I(i,j)+I(i,j+1)+I(i,j+2)+...
               I(i+1,j-2)+I(i+1,j-1)+I(i+1,j)+I(i+1,j+1)+I(i+1,j+2)+...
               I(i+2,j-2)+I(i+2,j-1)+I(i+2,j)+I(i+2,j+1)+I(i+2,j+2);
        if sum5 < AMF_THRESHOLD % Adaptive Threshold
            am_bw_amf(i,j)= round(sum5/25);
        else % Smaller 3x3 Adaptive Mean filter to save more details for dense area
            sum3 = I(i-1,j-1)+I(i-1,j)+I(i-1,j+1)+I(i,j-1)+I(i,j)+I(i,j+1)+I(i+1,j-1)+I(i+1,j)+I(i+1,j+1);
            am_bw_amf(i,j) = round(sum3/9);
        end
    end
end

% figure
% subplot(2,1,1),imshow(am_bw);title('Original');
% subplot(2,1,2),imshow(am_bw_amf);title('AMF');

% Mining Distance-Based Outliers in Large Datasets
% Reference: Knorr E M, Ng R T. Algorithms for Mining Distance-Based Outliers in Large Datasets[C]// International Conference on Very Large Data Bases. Morgan Kaufmann PLJblishers Inc. 1998:392¡ª403

J = am_bw_amf;
[m,n] = size(J);
am_rm_isolated = zeros(m,n);
i_rm = ceil(isolated_r/2); % index removed
for i = i_rm:m-i_rm
    for j = i_rm:n-i_rm
        count = 0;
        for k = -floor(isolated_r/2):floor(isolated_r/2) % scanning box
            for l = -floor(isolated_r/2):floor(isolated_r/2)
                if J(i+k,j+l) == 1
                    distance = round(sqrt(k^2+l^2));
                    if distance <= isolated_r
                        count = count + 1;
                    end
                end
            end
        end
        if count < ISOLATED_THRESHOLD % Isolating Threshold
            am_rm_isolated(i,j)= 0;
        else
            am_rm_isolated(i,j)= J(i,j);
        end
    end
end

if plot_switch == 1
    figure(8)
    subplot(3,1,1),imshow(am_bw);title('Ostu');
    subplot(3,1,2),imshow(am_bw_amf);title('AMF');
    subplot(3,1,3),imshow(am_rm_isolated);title('Removed isolated points');
end

% Comments: i.jogging and j.jumping forward generates abnormal data probably
% because of the hardware

% In-situ & non-in-situ classification using number of effective range bins

ERB = 0; % effective range bins

[m,n] = size(am_rm_isolated);
for i = 1:m
    for j = 1:n
        if am_rm_isolated(i,j) == 1
            effective_range_bins_array(i,(person_number-1)*60+(motion_label-1)*10+data_number) = 1;
            ERB = ERB + 1;
            break;
        end
    end
end

if loop_mode == 1
    effective_range_bins(1,(motion_label-1)*10+data_number) = ERB;
end
    
% ERB % 1 15 18 13 15 18 // 33 31 xx xx 31 35
if ERB < SITU_THRESHOLD
    motion_type = 1; % in-situ motion
else
    motion_type = 2; % non-in-situ motion
end

% ERB
if loop_mode == 1
    pred_type(1,(motion_label-1)*10+data_number) = motion_type;
    if person_number == 1
        save('pred_types1.mat','pred_type');
    elseif person_number == 2
        save('pred_types2.mat','pred_type');
    end
end
    


%% Feature extraction %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% STFT
% STFT of the single fast time row with the largest energy
mean_row=mean(am_median,2);
[~,index]=sort(mean_row,'descend');
am_max=am_median(index(1),:)';

fs = 100;
ts = 0:1/fs:4;

% [stft_am_max,F,T] = stft(am_max,fs,'Window',kaiser(64,5));%,'OverlapLength',220,'FFTLength',512); % Compute and plot the STFT of the signal. Use a Kaiser window of length 256 and shape parameter ¦Â=5. Specify the length of overlap as 220 samples and DFT length as 512 points. Plot the STFT with default colormap and view.
[stft_am_max,F,T] = stft(am_max,fs,'Window',hamming(16,'periodic'));%,'OverlapLength',220,'FFTLength',512); % Compute and plot the STFT of the signal. Use a Kaiser window of length 256 and shape parameter ¦Â=5. Specify the length of overlap as 220 samples and DFT length as 512 points. Plot the STFT with default colormap and view.

if stft_plot == 1 && plot_switch == 1
    figure(9)
    surf(T,F,abs(stft_am_max))
    shading interp
    view(0,90)
    axis tight
    colormap hsv
    colorbar
    title('STFT max Visualisation');
    xlabel('Time/s)');
    ylabel('Frequency/Hz');
end

%%% Another implementation of STFT: https://blog.csdn.net/lanchunhui/article/details/72240693

% figure;
% win_sz = 128;
% han_win = hanning(win_sz);
% 
% nfft = win_sz;
% nooverlap = win_sz - 1;
% [S, F, T] = spectrogram(am_max, window, nooverlap, nfft, fs);
% 
% imagesc(T, F, log10(abs(S)))
% set(gca, 'YDir', 'normal')
% xlabel('Time (secs)')
% ylabel('Freq (Hz)')
% title('short time fourier transform spectrum')

%%% STFT of averaging all fast time rows

% % [stft_am_max,F2,T2] = stft(am_max,fs,'Window',kaiser(64,5));%,'OverlapLength',220,'FFTLength',512); % Compute and plot the STFT of the signal. Use a Kaiser window of length 256 and shape parameter ¦Â=5. Specify the length of overlap as 220 samples and DFT length as 512 points. Plot the STFT with default colormap and view.
% am_mean = mean(am_median,1);
% [stft_am_mean,F2,T2] = stft(am_mean,fs,'Window',hamming(16,'periodic'));%,'OverlapLength',220,'FFTLength',512); % Compute and plot the STFT of the signal. Use a Kaiser window of length 256 and shape parameter ¦Â=5. Specify the length of overlap as 220 samples and DFT length as 512 points. Plot the STFT with default colormap and view.
% figure
% surf(T2,F2,abs(stft_am_mean))
% shading interp
% view(0,90)
% axis tight
% colormap hsv
% colorbar
% title('STFT Visualisation');
% xlabel('Time/s)');
% ylabel('Frequency/Hz');

if WINDOW_LENGTH == 16
    wrtft = zeros(16,96);
elseif WINDOW_LENGTH == 32
    wrtft = zeros(32,46);
end

%%%% In-situ motions feature extraction
if motion_type == 1
    
    %%% WRTFT % Ref:Erol B, Amin M G. Fall motion detection using combined range and Doppler features[C]//Signal Processing Conference.IEEE,2016:2075-2080. 
    power = am_median.^2;
    energy = mean(am_median,2);
    total_energy = sum(energy);
    weight = energy/total_energy;

    for i = 1:size(am_median,1)
        if effective_range_bins_array(i,(person_number-1)*60+(motion_label-1)*10+data_number) == 1
            [stft_tmp,F,T] = stft(am_median(i,:),fs,'Window',hamming(WINDOW_LENGTH,'periodic'));%,'OverlapLength',220,'FFTLength',512); % Compute and plot the STFT of the signal. Use a Kaiser window of length 256 and shape parameter ¦Â=5. Specify the length of overlap as 220 samples and DFT length as 512 points. Plot the STFT with default colormap and view.
            wrtft = wrtft + 1*weight(i)*stft_tmp;
        end
    end

    wrtft_amp = abs(wrtft);

    if  plot_switch == 1
        figure(10)
        surf(T,F,wrtft_amp)
        shading interp
        view(0,90)
        axis tight
        colormap hsv
        colorbar
        title('In-situ WRTFT Visualisation');
        xlabel('Time/s)');
        ylabel('Frequency/Hz');
        hold on
    end
    
    %%% Calculate envelope of torso motion
%     if WINDOW_LENGTH == 16
%         torso = zeros(16,96);
%     elseif WINDOW_LENGTH == 32
%         torso = zeros(32,46);
%     end


    F_env = zeros(size(wrtft_amp,2),1);
    for j = 1:size(wrtft_amp,2)
    %     envlope_threshold = ALPHA*sum(wrtft_amp(:,j)); % percentage threshold
        envlope_threshold = TORSO_ENERGY; % fixed threshold

        env_find = 0;

        for i = 1:size(wrtft_amp,1)

            if wrtft_amp(i,j) >= envlope_threshold
                if plot_switch == 1
                    plot3(T(j),F(WINDOW_LENGTH+1-i),wrtft_amp(i,j),'.b','markersize',10)
                end
%                 torso(i,j) = 1;
                F_env(j) = F(WINDOW_LENGTH+1-i); % Envelope frequencies
                env_find = 1; % if envolope is found
                break;
            end
        end
        if env_find == 0
            if plot_switch == 1
                plot3(T(j),F(floor((1/2)*WINDOW_LENGTH)),wrtft_amp(floor((1/2)*WINDOW_LENGTH),j),'.b','markersize',10)
            end
            F_env(j) = 0;
        end
    end

    %%% Extract Physical Empirical Features

    % 1. Max Envelope Frequency: Fmax
    if loop_mode == 1
        Fmax(1,(motion_label-1)*10+data_number) = max(F_env);
    end
    % 2. Average Envelope Frequency (using sliding window): avg_env_freq
    % 3. Variances of Envelope Frequencies (using sliding window): var_env_freq
    % Ref: https://wenku.baidu.com/view/f0d83841a8956bec0975e320.html Chapter 4.3.1
    SLIDING_WINDOW_LENGTH = 20;
    SLIDING_OVERLAP = 10;
    step = SLIDING_WINDOW_LENGTH - SLIDING_OVERLAP;
    avg_var_length = ceil((size(F_env,1)+step)/step);
    avg_env_freq = zeros(1,avg_var_length);
    var_env_freq = zeros(1,avg_var_length);
    mean_index = 1;

    for i = 1:step:(size(F_env,1)+step)
        counter = SLIDING_WINDOW_LENGTH;
        sum_env_freq = 0;
        varN_env_freq = 0;
        for k = i + 1 - ceil(SLIDING_WINDOW_LENGTH/2) : i + floor(SLIDING_WINDOW_LENGTH/2)
            if k <= 0 || k > size(F_env,1)
                counter = counter - 1;
            else
                sum_env_freq = sum_env_freq + F_env(k,1);
            end
        end

        avg_env_freq(1,mean_index) = sum_env_freq/counter;

        counter = SLIDING_WINDOW_LENGTH;

        for k = i + 1 - ceil(SLIDING_WINDOW_LENGTH/2) : i + floor(SLIDING_WINDOW_LENGTH/2)
            if k <= 0 || k > size(F_env,1)
                counter = counter - 1;
            else
                varN_env_freq = varN_env_freq + (F_env(k,1)-avg_env_freq(1,mean_index))^2;
            end
        end

        var_env_freq(1,mean_index) = varN_env_freq/counter;

        mean_index = mean_index + 1;
    end
    
    if loop_mode == 1
        torso_mean(:,(motion_label-1)*10+data_number) = avg_env_freq';
        torso_var(:,(motion_label-1)*10+data_number) = var_env_freq';
    end
    % 4. Number of Effective Range Bins (already calculated: effective_range_bins)
    

%%%% Non-in-situ motions feature extraction
elseif motion_type == 2
    
    %%% WRTFT with weights being all 1
    
    for i = 1:size(am_median,1)
        [stft_tmp,F,T] = stft(am_median(i,:),fs,'Window',hamming(WINDOW_LENGTH,'periodic'));%,'OverlapLength',220,'FFTLength',512); % Compute and plot the STFT of the signal. Use a Kaiser window of length 256 and shape parameter ¦Â=5. Specify the length of overlap as 220 samples and DFT length as 512 points. Plot the STFT with default colormap and view.
        wrtft = wrtft + stft_tmp;
    end

    wrtft_amp = abs(wrtft);
    
    if plot_switch == 1
        figure(11)
        surf(T,F,wrtft_amp)
        shading interp
        view(0,90)
        axis tight
        colormap hsv
        colorbar
        title('Non-in-situ WRTFT Visualisation');
        xlabel('Time/s)');
        ylabel('Frequency/Hz');
    end
    
    %%% Principle Component Analysis (PCA)
    % Ref: Imperial EEE Pattern Recognition course
    
    X = wrtft_amp;
    Nt = size(X,2);
    Nf = size(X,1);
    M = 5;
    X_bar = mean(X,2);
    A = X - X_bar;
    
    ATA = transpose(A) * A / Nt; % Covariance matrix in low dimension
    [Evec_raw_ld,Eval_raw_ld] = eig(ATA); % ld stands for low dimension

    Evec_ld = fliplr(Evec_raw_ld);
    Eval_ld = fliplr(flipud(Eval_raw_ld));
    
    Eval_ld_diag = diag(Eval_ld);
    Eval_ld_sum = sum(Eval_ld_diag);
    Eval_ld_accumulated_energy_ratio = zeros(size(Eval_ld_sum,2),1);
    
    number_of_eigenvalue_occupy_99_energy_find = 0;
    
    Eval_ld_accumulated_energy_ratio(1) = abs(Eval_ld_diag(1))/Eval_ld_sum;
    for i = 2:size(Eval_ld_diag,1)
        Eval_ld_accumulated_energy_ratio(i) = Eval_ld_accumulated_energy_ratio(i-1) + abs(Eval_ld_diag(i))/Eval_ld_sum;
        if Eval_ld_accumulated_energy_ratio(i) >= 0.99 && number_of_eigenvalue_occupy_99_energy_find == 0
            number_of_eigenvalue_occupy_99_energy = i;
            number_of_eigenvalue_occupy_99_energy_find = 1;
        end
    end
    
%     number_of_eigenvalue_occupy_99_energy;
    if plot_switch == 1
        figure(12)
        plot(abs(Eval_ld_diag),'b-')
        yyaxis left
        title('Plots with Different y-Scales')
        xlabel('Sorted Eigenvalue Number')
        ylabel('Eigenvalue')
        hold on;
        yyaxis right
        ylabel('Accumulated Energy Ratio')
        plot(Eval_ld_accumulated_energy_ratio,'r-')
    end
    
    Evec_best_ld = Evec_ld(1:Nt,1:M);
    Eval_best_ld = Eval_ld(1:M,1:M);
    Evec_best_ld = normc(A * Evec_best_ld);

    % a is the column space of Wn
    a = transpose(transpose(A) * Evec_best_ld);
    
    % non-in-situ feature
%     non_in_situ_feature_e(:,(motion_label-1)*10+data_number-60) = Evec_best_ld(:);
    if loop_mode == 1
        non_in_situ_feature_a(:,(person_number-1)*60+(motion_label-1)*10+data_number-60) = a(:);
    end
end

    end
end


true_type(1,1:60) = repelem(1,60);
true_type(1,61:120) = repelem(2,60);

same_type = (true_type == pred_type);
type_classify_accuracy = sum(same_type)/120

% in-situ feature

in_situ_feature(1:6,1+(person_number-1)*60:60+(person_number-1)*60) = repmat(Fmax,6,1);
in_situ_feature(7:12,1+(person_number-1)*60:60+(person_number-1)*60) = torso_mean;
in_situ_feature(13:18,1+(person_number-1)*60:60+(person_number-1)*60) = torso_var;
in_situ_feature(19:24,1+(person_number-1)*60:60+(person_number-1)*60) = repmat(effective_range_bins(1,1:60),6,1);
in_situ_feature(25,1+(person_number-1)*60:60+(person_number-1)*60) = label(1,1:60);


end

% non-in-situ feature
% non_in_situ_feature_e(513,:) = label(1,61:120);
for person_number = 1:PERSON
    non_in_situ_feature_a(231,1+(person_number-1)*60:60+(person_number-1)*60) = label(1,61:120);
end