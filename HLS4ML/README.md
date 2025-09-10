- Sistema operativo:    LINUX 24.04.2-> WSL DE WINDOWS
- La versión de python y librerias están en environment_HLS4ML.yml
- Para conversion de .h5 (modelo machine learning normal) para generar los scripts de C++ para poder utilizarlos en VIVADO HLS. --> conversion_model_with_hls4ml.py
- Si todo va bien, se genera automáticamente la IP.
- En los archivos que se generan, busco la carpeta myproject_prj, ahi, solution1/impl/ip/ xilinx_com_hls_myproject.zip
- Dentro de este .zip está la IP

  - A partir de este momento habrimos Vivado 2019.2 y seguimos las instrucciones del video.


  - OJO problema con Vivado 2019.2
  - Si VIDADO HLS da problema buscar y2k22_patch. Dentro de la carpeta readme con explicación.
  - En el foro de Xilinx solucion 
