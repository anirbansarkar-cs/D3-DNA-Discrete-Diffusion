import subprocess
import glob

#Run modisco lite
# for dataset in glob.glob('./attr_analysis/saliency_score/*.npz'):
#     saliency_f = dataset
#     dataset = dataset.split('/')[-1]
#     seq_f = './attr_analysis/saliency_seq/' + dataset
    
#     cmd = 'modisco motifs -s %s -a %s -n 2000 -w 248 -o ./modisco_analysis/%s.h5'%(seq_f,saliency_f,dataset[:-4])
#     print(cmd)
#     subprocess.call(cmd,shell=True)
    
    
#Generate report
for result in glob.glob('./modisco_analysis/*.h5'):
    dataset = result.split('/')[-1][:-3]
    cmd = 'modisco report -i %s -o ./modisco_analysis/%s_report/ -s ./modisco_analysis/%s_report/ -m ./modisco_analysis/JASPAR2022_CORE_vertebrates.meme'%(result,dataset,dataset)
    print(cmd)
    subprocess.call(cmd,shell=True)
