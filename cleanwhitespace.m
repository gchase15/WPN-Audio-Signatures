clear all; close all; clc;
% load 'Gavin_WeaponSounds/AK47/*.wav';

files = dir('Gavin_WeaponSounds/FN FAL/*.wav') ; 


minWhitespace = 5;
N = length(files);   
for i = 1:N
    test = [];
    consecutiveWhitespace = [];
    thisfile = files(i).name ; 
    [data,fs] = audioread(thisfile);
%     sound(data, fs);
    datalength = length(data);
    for j = 1:datalength
        if (j > datalength)
            break;
        end
        if (data(j,1) ~= 0)
            consecutiveWhitespace = [];
            continue;
        else
            consecutiveWhitespace = [consecutiveWhitespace data(j,1)];
        end
        whitespacesize = size(consecutiveWhitespace,2);
        if (whitespacesize == minWhitespace)
            for k=1:whitespacesize
                data(j+1-k,:) = [];
                if (j+1-k > length(data(:, 1)))
                    break;
                end
                vtest = [test data(j+1-k,1)];
                datalength = datalength - 1;
            end
            consecutiveWhitespace = [];
        end
    end
    newFileName = split(thisfile,".");
    filename = string(newFileName(1,1));
    newFileNameChar = filename + 'Cleaned.wav';
    audiowrite(newFileNameChar, data, fs);

end
