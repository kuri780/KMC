本文件夹中共有两个文件
两个文件均为KMC（蒙特卡洛法）模拟二维薄膜生长
其中main文件在遍历时会更新所有位点的n值（其中n代表该位点周围的粒子数量）
KMC_Metropolis文件在原文件基础上进行增量计算的优化，每次遍历只会修改发生吸附、脱附或者迁移的粒子周围位点的n值，减少了计算时间
