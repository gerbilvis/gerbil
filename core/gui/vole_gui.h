#ifndef VOLE_GUI_H
#define VOLE_GUI_H


#include "ui_vole_gui.h"

#include "vole_user_paths.h"

#include <QtGui/QMainWindow>

#include <QMenu>
#include <QSignalMapper>
#include <vector>


class CommandWrapper;

class VoleGui : public QMainWindow, private Ui::VoleGui
{
	Q_OBJECT

public:
	VoleGui(QWidget *parent = 0);
	QTabWidget* container() { return workspaceContainer; }
	~VoleGui();
	
	void remove(CommandWrapper* wrapper);

	vole::UserPaths userPaths;
	
public slots:
	void spawn(int id);
	void removeTab(bool);

private:
	QPushButton *cornerWidget;
	typedef CommandWrapper* (*startFunc)(VoleGui*);
	
	bool eventFilter(QObject*, QEvent*);
	
	void registerMethod(const QString& name, startFunc spawn);
	
	// popup menu
	std::vector<std::pair<QAction*, startFunc> > startActions;
	QSignalMapper *startMap;
	QMenu startMenu;

	// running workplaces
	std::vector<CommandWrapper*> wrappers;

};

#endif // VOLE_GUI_H
