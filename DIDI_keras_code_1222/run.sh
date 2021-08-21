python rule_demo.py
cd link_fea
python link2graph.py --feature-type=time 
python graph2vec.py --feature-type=time
cd ..
python features.py
python w2v_cross.py
python simp_nn1.py
python simp_nn2.py