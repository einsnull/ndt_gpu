#include <fast_pcl/ndt_gpu/NormalDistributionsTransform.h>


#include <pcl/point_cloud.h>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
//#include <pcl_conversions/pcl_conversions.h>
#include <pcl/registration/ndt.h>
#include <pcl/filters/approximate_voxel_grid.h>

#include <pcl/visualization/pcl_visualizer.h>
#include <boost/thread/thread.hpp>

#include <time.h> 

#include <iostream>



int main(int argc, char** argv)
{
  if(argc < 3){
    std::cout << "Too few params" << std::endl;
    return -1;
  }

  bool use_gpu = false;
  if(argc == 4){
    use_gpu = true;
  }

  // Loading first scan of room.
  pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud (new pcl::PointCloud<pcl::PointXYZ>);
  if (pcl::io::loadPCDFile<pcl::PointXYZ> (argv[1], *target_cloud) == -1)
  {
    PCL_ERROR ("Couldn't read file room_scan1.pcd \n");
    return (-1);
  }
  std::cout << "Loaded " << target_cloud->size () << " data points from room_scan1.pcd" << std::endl;

  // Loading second scan of room from new perspective.
  pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud (new pcl::PointCloud<pcl::PointXYZ>);
  if (pcl::io::loadPCDFile<pcl::PointXYZ> (argv[2], *input_cloud) == -1)
  {
    PCL_ERROR ("Couldn't read file room_scan2.pcd \n");
    return (-1);
  }
  std::cout << "Loaded " << input_cloud->size () << " data points from room_scan2.pcd" << std::endl;

  // Filtering input scan to roughly 10% of original size to increase speed of registration.
  pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::ApproximateVoxelGrid<pcl::PointXYZ> approximate_voxel_filter;
  approximate_voxel_filter.setLeafSize (0.2, 0.2, 0.2);
  approximate_voxel_filter.setInputCloud (input_cloud);
  approximate_voxel_filter.filter (*filtered_cloud);
  std::cout << "Filtered cloud contains " << filtered_cloud->size ()
            << " data points from room_scan2.pcd" << std::endl;

  // Initializing Normal Distributions Transform (NDT).
  pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> ndt;
  //gpu::GNormalDistributionsTransform ndt;

  // Setting scale dependent NDT parameters
  // Setting minimum transformation difference for termination condition.
  ndt.setTransformationEpsilon (0.01);
  // Setting maximum step size for More-Thuente line search.
  ndt.setStepSize (0.1);
  //Setting Resolution of NDT grid structure (VoxelGridCovariance).
  ndt.setResolution (1.0);

  // Setting max number of registration iterations.
  ndt.setMaximumIterations (35);

  // Setting point cloud to be aligned.
  ndt.setInputSource (filtered_cloud);
  // Setting point cloud to be aligned to.
  ndt.setInputTarget (target_cloud);

  // std::shared_ptr<gpu::GNormalDistributionsTransform> new_gpu_ndt_ptr = std::make_shared<gpu::GNormalDistributionsTransform>();
  // new_gpu_ndt_ptr->setResolution(1.0);
  // new_gpu_ndt_ptr->setInputTarget(target_cloud);
  // new_gpu_ndt_ptr->setMaximumIterations(35);
  // new_gpu_ndt_ptr->setStepSize(0.1);
  // new_gpu_ndt_ptr->setTransformationEpsilon(0.01);
  // new_gpu_ndt_ptr->setInputSource(filtered_cloud);

  gpu::GNormalDistributionsTransform g_ndt;

  // Setting scale dependent NDT parameters
  // Setting minimum transformation difference for termination condition.
  g_ndt.setTransformationEpsilon (0.01);
  // Setting maximum step size for More-Thuente line search.
  g_ndt.setStepSize (0.1);
  //Setting Resolution of NDT grid structure (VoxelGridCovariance).
  g_ndt.setResolution (1.0);

  // Setting max number of registration iterations.
  g_ndt.setMaximumIterations (35);

  // Setting point cloud to be aligned.
  g_ndt.setInputSource (filtered_cloud);
  // Setting point cloud to be aligned to.
  g_ndt.setInputTarget (target_cloud);


  // Set initial alignment estimate found using robot odometry.
  Eigen::AngleAxisf init_rotation (0.6931, Eigen::Vector3f::UnitZ ());
  Eigen::Translation3f init_translation (1.79387, 0.720047, 0);
  Eigen::Matrix4f init_guess = (init_translation * init_rotation).matrix ();

  // Calculating required rigid transform to align the input cloud to the target cloud.
  pcl::PointCloud<pcl::PointXYZ>::Ptr output_cloud (new pcl::PointCloud<pcl::PointXYZ>);
  clock_t start = clock();
  Eigen::Matrix4f final_trans;
  bool converged = false;
  double fitness_score = -1;

  if(use_gpu){
    // new_gpu_ndt_ptr->align(init_guess);
    // final_trans = new_gpu_ndt_ptr->getFinalTransformation();
    // converged = new_gpu_ndt_ptr->hasConverged();
    // fitness_score = new_gpu_ndt_ptr->getFitnessScore();
    g_ndt.align(init_guess);
    final_trans = g_ndt.getFinalTransformation();
    //final_trans = init_guess;
    converged = g_ndt.hasConverged();
    fitness_score = g_ndt.getFitnessScore();

  }
  else{
    ndt.align (*output_cloud, init_guess);
    final_trans = ndt.getFinalTransformation ();
    converged = ndt.hasConverged();
    fitness_score = ndt.getFitnessScore();
  }

  clock_t finish = clock();
  double duration = (double)(finish - start) / CLOCKS_PER_SEC;
  std::cout << "Time cost:" << duration << std::endl;

  std::cout << "Normal Distributions Transform has converged:" << converged
            << " score: " << fitness_score << std::endl;

  // Transforming unfiltered, input cloud using found transform.
  pcl::transformPointCloud (*input_cloud, *output_cloud, final_trans);

  // Saving transformed input cloud.
  pcl::io::savePCDFileASCII ("out.pcd", *output_cloud);

  // Initializing point cloud visualizer
  boost::shared_ptr<pcl::visualization::PCLVisualizer>
  viewer_final (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer_final->setBackgroundColor (0, 0, 0);

  // Coloring and visualizing target cloud (red).
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
  target_color (target_cloud, 255, 0, 0);
  viewer_final->addPointCloud<pcl::PointXYZ> (target_cloud, target_color, "target cloud");
  viewer_final->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
                                                  1, "target cloud");

  // Coloring and visualizing transformed input cloud (green).
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
  output_color (output_cloud, 0, 255, 0);
  viewer_final->addPointCloud<pcl::PointXYZ> (output_cloud, output_color, "output cloud");
  viewer_final->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
                                                  1, "output cloud");

  // Starting visualizer
  viewer_final->addCoordinateSystem (1.0, "global");
  viewer_final->initCameraParameters ();

  // Wait until visualizer window is closed.
  while (!viewer_final->wasStopped ())
  {
    viewer_final->spinOnce (100);
    boost::this_thread::sleep (boost::posix_time::microseconds (100000));
  }


  return 0;
}
