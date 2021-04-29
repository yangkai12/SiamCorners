ft = open("/data/study/code/pysot-master01/tools/hp_search_result/VOT2018/checkpoint_e6_batchsize56_r255_pk-0.050_wi-0.100_lr-0.350/baseline/ants1", 'w')
for entry in result :
    print(entry)
try:
    ft.write(entry+'\n')
except:
    log.error('write backup error:'+JOBNAME)
ft.close()#在内容写完后再关闭文件
os.chdir(basePath)