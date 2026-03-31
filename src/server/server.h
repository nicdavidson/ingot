#ifndef INGOT_SERVER_H
#define INGOT_SERVER_H

#include "model/model.h"

// Start the HTTP server. Blocks until shutdown.
// Returns 0 on clean shutdown, -1 on error.
int server_run(Model *model, int port);

#endif
