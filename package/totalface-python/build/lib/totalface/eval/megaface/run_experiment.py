# Created by MegaFace Team
# Please cite the our paper if you use our code, results, or dataset in a publication
# http://megaface.cs.washington.edu/ 

import argparse
import os
import sys
import json
import random
import subprocess

def run_experiment_main(jb_model,exe_id,exe_fuse,magaface_lst_json,prob_lst_json,features_mega,features_prob,file_ending,save_base,sizes,num_sets,dm):

    distractor_feature_path = features_mega
    out_root = save_base
    probe_feature_path = features_prob
    model = jb_model
    num_sets = num_sets
    
    file_ending = file_ending
    alg_name = file_ending.split('.')[0].strip('_')
    delete_matrices = dm
    probe_list_basename = prob_lst_json
    megaface_list_basename = os.path.join(os.path.dirname(magaface_lst_json),os.path.basename(magaface_lst_json))
    set_indices = range(1,int(num_sets) + 1)

    print(distractor_feature_path)
    assert os.path.exists(distractor_feature_path)
    assert os.path.exists(probe_feature_path)
    if not os.path.exists(out_root):
        os.makedirs(out_root)
    if(not os.path.exists(os.path.join(out_root, "otherFiles"))):
        os.makedirs(os.path.join(out_root, "otherFiles"))
    other_out_root = os.path.join(out_root, "otherFiles")

    probe_name = os.path.basename(probe_list_basename).split('_')[0]
    distractor_name = os.path.basename(megaface_list_basename).split('_')[0]

    #Create feature lists for megaface for all sets and sizes and verifies all features exist
    missing = False
    for index in set_indices:
        for size in sizes:
            print('Creating feature list of {} photos for set {}'.format(size,str(index)))
            cur_list_name = megaface_list_basename + "_{}_{}".format(str(size), str(index))
            with open(cur_list_name) as fp:
                featureFile = json.load(fp)
                path_list = featureFile["path"]
                for i in range(len(path_list)):
                    #use_path = path_list[i].split("/")[-2]+"/"+path_list[i].split("/")[-1]
                    #path_list[i] = os.path.join(distractor_feature_path,use_path + file_ending)
                    path_list[i] = os.path.join(distractor_feature_path,path_list[i] + file_ending)
                    if(not os.path.isfile(path_list[i])):
                        print(path_list[i] + " is missing")
                        missing = True
                    if (i % 10000 == 0 and i > 0):
                        print(str(i) + " / " + str(len(path_list)))
                featureFile["path"] = path_list
                json.dump(featureFile, open(os.path.join(
                    other_out_root, '{}_features_{}_{}_{}'.format(distractor_name,alg_name,size,index)), 'w'), sort_keys=True, indent=4)
    if(missing):
        sys.exit("Features are missing...")
    
    #Create feature list for probe set
    with open(probe_list_basename) as fp:
        featureFile = json.load(fp)
        path_list = featureFile["path"]
        for i in range(len(path_list)):
            path_list[i] = os.path.join(probe_feature_path,path_list[i] + file_ending)
            if(not os.path.isfile(path_list[i])):
                print(path_list[i] + " is missing")
                missing = True
        featureFile["path"] = path_list
        json.dump(featureFile, open(os.path.join(
            other_out_root, '{}_features_{}'.format(probe_name,alg_name)), 'w'), sort_keys=True, indent=4)
        probe_feature_list = os.path.join(other_out_root, '{}_features_{}'.format(probe_name,alg_name))
    if(missing):
        sys.exit("Features are missing...")

    print('Running probe to probe comparison')
    probe_score_filename = os.path.join(
        other_out_root, '{}_{}_{}.bin'.format(probe_name, probe_name, alg_name))
    proc = subprocess.Popen(
        [exe_id, model, "path", probe_feature_list, probe_feature_list, probe_score_filename])
    proc.communicate()

    for index in set_indices:
        for size in sizes:
            print('Running test with size {} images for set {}'.format(
                str(size), str(index)))
            args = [exe_id, model, "path", os.path.join(other_out_root, '{}_features_{}_{}_{}'.format(distractor_name,alg_name,size,index)
                ), probe_feature_list, os.path.join(other_out_root, '{}_{}_{}_{}_{}.bin'.format(probe_name, distractor_name, alg_name, str(size),str(index)))]
            proc = subprocess.Popen(args)
            proc.communicate()

            print('Computing test results with {} images for set {}'.format(
                str(size), str(index)))
            args = [exe_fuse]
            args += [os.path.join(other_out_root, '{}_{}_{}_{}_{}.bin'.format(
                probe_name, distractor_name, alg_name, str(size), str(index)))]
            args += [os.path.join(other_out_root, '{}_{}_{}.bin'.format(
                probe_name, probe_name, alg_name)), probe_feature_list, str(size)]
            args += [os.path.join(out_root, "cmc_{}_{}_{}_{}_{}.json".format(
                probe_name, distractor_name, alg_name, str(size), str(index)))]
            args += [os.path.join(out_root, "matches_{}_{}_{}_{}_{}.json".format(
                probe_name, distractor_name, alg_name, str(size), str(index)))]
            proc = subprocess.Popen(args)
            proc.communicate()

            if(delete_matrices):
                os.remove(os.path.join(other_out_root, '{}_{}_{}_{}_{}.bin'.format(
                    probe_name, distractor_name, alg_name, str(size), str(index))))

