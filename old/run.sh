#!/bin/bash

for i in {4..4}
do
echo '#!/bin/bash' > submit_job
echo "#SBATCH -t 02:00:00">> submit_job
echo "#SBATCH --partition=chihway" >> submit_job
echo "#SBATCH --account=pi-chihway" >> submit_job
echo "#SBATCH --exclusive" >> submit_job
echo "#SBATCH --nodes=1" >> submit_job
echo "#SBATCH --job-name=healqest-${i}" >> submit_job
echo "cd $PWD" >> submit_job
echo "source /home/yomori/setup.sh" >> submit_job
echo python reconsructlens.py ${i} ${i} >> submit_job
sbatch submit_job
done

