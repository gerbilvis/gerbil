vole_module_name("gerbil_common")
vole_module_description("Common classes used in gerbil")
vole_module_variable("Gerbil_Common")

vole_add_required_dependencies("OPENCV" "BOOST_DATE_TIME" "TBB")
vole_add_optional_dependencies("BOOST_SYSTEM" "BOOST_THREAD" "BOOST_FILESYSTEM" "QT")

vole_compile_library(
	multi_img/multi_img_ext
	multi_img/multi_img_io_ext
    multi_img/multi_img_offloaded
    multi_img/multi_img_tbb
	multi_img/illuminant
    background_task/background_task
    background_task/background_task_queue
    background_task/tasks/multi_img_tasks
    shared_data
	rectangles
)

if(QT_FOUND)
vole_moc_library(
	background_task/background_task.h
)
endif()

vole_add_module()
