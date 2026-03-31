#ifndef INGOT_ROUTES_H
#define INGOT_ROUTES_H

#include "model/model.h"
#include "server/request_parse.h"

// Handle an HTTP request and write the response to fd.
void route_handle(int fd, const HttpRequest *req, Model *model);

#endif
