#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int nx, ny, px, py;
    double xlow, xhigh, ylow, yhigh;
    double t0, t_end, kappa_x, kappa_y;
    double T_initial;

    /* Process 0 reads the input file and broadcasts the parameters */
    if (rank == 0) {
        FILE *fp = fopen("input2d.in", "r");
        if (fp == NULL) {
            perror("Error opening input2d.in");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        fscanf(fp, "%d %d", &nx, &ny);
        fscanf(fp, "%lf %lf %lf %lf", &xlow, &xhigh, &ylow, &yhigh);
        fscanf(fp, "%lf %lf %lf %lf", &t0, &t_end, &kappa_x, &kappa_y);
        fscanf(fp, "%lf", &T_initial);
        fscanf(fp, "%d %d", &px, &py);
        fclose(fp);
    }

    MPI_Bcast(&nx, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ny, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&xlow, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&xhigh, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ylow, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&yhigh, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&t0, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&t_end, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&kappa_x, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&kappa_y, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&T_initial, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&px, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&py, 1, MPI_INT, 0, MPI_COMM_WORLD);

    /* Verify that the number of processes matches the specified process grid */
    if (size != px * py) {
        if (rank == 0)
            fprintf(stderr, "Error: Number of MPI processes (%d) does not match px*py (%d)\n", size, px*py);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    /* Create a 2D Cartesian communicator */
    int dims[2] = {px, py};
    int periods[2] = {0, 0}; // Non-periodic boundaries
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &cart_comm);

    int coords[2];
    MPI_Cart_coords(cart_comm, rank, 2, coords);
    int proc_x = coords[0]; // Process coordinate in x-direction
    int proc_y = coords[1]; // Process coordinate in y-direction

    /* Determine the local grid size */
    int local_nx = nx / px;
    int local_ny = ny / py;

    /* Compute the global grid spacings */
    double dx = (xhigh - xlow) / (nx - 1);
    double dy = (yhigh - ylow) / (ny - 1);

    /* Determine the time step from the CFL condition */
    double dt_x = dx * dx / kappa_x;
    double dt_y = dy * dy / kappa_y;
    double dt = 0.25 * (dt_x < dt_y ? dt_x : dt_y);

    /* Allocate local arrays with ghost layers */
    int ldim = local_nx + 2;
    int lsize = (local_ny + 2) * ldim;
    double *T_old = (double*) malloc(lsize * sizeof(double));
    double *T_new = (double*) malloc(lsize * sizeof(double));

    if (!T_old || !T_new) {
        fprintf(stderr, "Error allocating local arrays on process %d\n", rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    /* Initialize local arrays */
    int global_i_start = proc_x * local_nx;
    int global_j_start = proc_y * local_ny;

    for (int j = 1; j <= local_ny; j++) {
        int global_j = global_j_start + j - 1;
        double y = ylow + global_j * dy;
        for (int i = 1; i <= local_nx; i++) {
            int global_i = global_i_start + i - 1;
            double x = xlow + global_i * dx;
            if (global_i == 0 || global_i == nx - 1 || global_j == 0 || global_j == ny - 1) {
                T_old[j*ldim + i] = 0.0;
            } else if (x >= 0.4 && x <= 0.6 && y >= 0.4 && y <= 0.6) {
                T_old[j*ldim + i] = T_initial;
            } else {
                T_old[j*ldim + i] = 0.0;
            }
        }
    }

    /* Create MPI datatypes for halo exchange */
    MPI_Datatype column_type, row_type;
    MPI_Type_vector(local_ny, 1, ldim, MPI_DOUBLE, &column_type);
    MPI_Type_contiguous(local_nx, MPI_DOUBLE, &row_type);
    MPI_Type_commit(&column_type);
    MPI_Type_commit(&row_type);

    /* Determine neighbor ranks for halo exchanges */
    int north, south, east, west;
    MPI_Cart_shift(cart_comm, 1, 1, &west, &east);
    MPI_Cart_shift(cart_comm, 0, 1, &north, &south);

    /* Time-stepping loop */
    double t = t0;
    int iter = 0;
    MPI_Request reqs[8];
    MPI_Status stats[8];

    double start_time = MPI_Wtime();

    while (t < t_end) {
        int req_count = 0;

        /* Start non-blocking halo exchanges */
        if (west != MPI_PROC_NULL) {
            MPI_Irecv(&T_old[1], 1, column_type, west, 0, cart_comm, &reqs[req_count++]);
            MPI_Isend(&T_old[ldim + 1], 1, column_type, west, 1, cart_comm, &reqs[req_count++]);
        }
        if (east != MPI_PROC_NULL) {
            MPI_Irecv(&T_old[ldim + local_nx], 1, column_type, east, 1, cart_comm, &reqs[req_count++]);
            MPI_Isend(&T_old[ldim + local_nx - 1], 1, column_type, east, 0, cart_comm, &reqs[req_count++]);
        }
        if (north != MPI_PROC_NULL) {
            MPI_Irecv(&T_old[local_nx + 1], 1, row_type, north, 2, cart_comm, &reqs[req_count++]);
            MPI_Isend(&T_old[ldim + 1], 1, row_type, north, 3, cart_comm, &reqs[req_count++]);
        }
        if (south != MPI_PROC_NULL) {
            MPI_Irecv(&T_old[(local_ny)*ldim + 1], 1, row_type, south, 3, cart_comm, &reqs[req_count++]);
            MPI_Isend(&T_old[(local_ny - 1)*ldim + 1], 1, row_type, south, 2, cart_comm, &reqs[req_count++]);
        }

        /* Update interior points */
        for (int j = 2; j <= local_ny - 1; j++) {
            for (int i = 2; i <= local_nx - 1; i++) {
                int idx = j * ldim + i;
                double d2Tdx2 = (T_old[j*ldim + (i+1)] - 2.0 * T_old[idx] + T_old[j*ldim + (i-1)]) / (dx*dx);
                double d2Tdy2 = (T_old[(j+1)*ldim + i] - 2.0 * T_old[idx] + T_old[(j-1)*ldim + i]) / (dy*dy);
                T_new[idx] = T_old[idx] + dt * (kappa_x * d2Tdx2 + kappa_y * d2Tdy2);
            }
        }

        /* Wait for halo exchanges to complete */
        MPI_Waitall(req_count, reqs, stats);

        /* Update boundary points */
        for (int j = 1; j <= local_ny; j++) {
            int i = 1;
            int idx = j * ldim + i;
            double d2Tdx2 = (T_old[j*ldim + (i+1)] - 2.0 * T_old[idx] + T_old[j*ldim + (i-1)]) / (dx*dx);
            double d2Tdy2 = (T_old[(j+1)*ldim + i] - 2.0 * T_old[idx] + T_old[(j-1)*ldim + i]) / (dy*dy);
            T_new[idx] = T_old[idx] + dt * (kappa_x * d2Tdx2 + kappa_y * d2Tdy2);

            i = local_nx;
            idx = j * ldim + i;
            d2Tdx2 = (T_old[j*ldim + (i+1)] - 2.0 * T_old[idx] + T_old[j*ldim + (i-1)]) / (dx*dx);
            d2Tdy2 = (T_old[(j+1)*ldim + i] - 2.0 * T_old[idx] + T_old[(j-1)*ldim + i]) / (dy*dy);
            T_new[idx] = T_old[idx] + dt * (kappa_x * d2Tdx2 + kappa_y * d2Tdy2);
        }

        for (int i = 1; i <= local_nx; i++) {
            int j = 1;
            int idx = j * ldim + i;
            double d2Tdx2 = (T_old[j*ldim + (i+1)] - 2.0 * T_old[idx] + T_old[j*ldim + (i-1)]) / (dx*dx);
            double d2Tdy2 = (T_old[(j+1)*ldim + i] - 2.0 * T_old[idx] + T_old[(j-1)*ldim + i]) / (dy*dy);
            T_new[idx] = T_old[idx] + dt * (kappa_x * d2Tdx2 + kappa_y * d2Tdy2);

            j = local_ny;
            idx = j * ldim + i;
            d2Tdx2 = (T_old[j*ldim + (i+1)] - 2.0 * T_old[idx] + T_old[j*ldim + (i-1)]) / (dx*dx);
            d2Tdy2 = (T_old[(j+1)*ldim + i] - 2.0 * T_old[idx] + T_old[(j-1)*ldim + i]) / (dy*dy);
            T_new[idx] = T_old[idx] + dt * (kappa_x * d2Tdx2 + kappa_y * d2Tdy2);
        }

        /* Reapply global Dirichlet boundary conditions */
        if (global_i_start == 0) {
            for (int j = 1; j <= local_ny; j++)
                T_new[j*ldim + 1] = 0.0;
        }
        if (global_i_start + local_nx == nx) {
            for (int j = 1; j <= local_ny; j++)
                T_new[j*ldim + local_nx] = 0.0;
        }
        if (global_j_start == 0) {
            for (int i = 1; i <= local_nx; i++)
                T_new[1*ldim + i] = 0.0;
        }
        if (global_j_start + local_ny == ny) {
            for (int i = 1; i <= local_nx; i++)
                T_new[local_ny*ldim + i] = 0.0;
        }

        /* Swap old and new arrays */
        double *temp_ptr = T_old;
        T_old = T_new;
        T_new = temp_ptr;

        t += dt;
        iter++;
    }

    double end_time = MPI_Wtime();
    double local_time = end_time - start_time;
    double max_time;
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("MPI computation completed in %d iterations\n", iter);
        printf("Total time: %f seconds\n", max_time);
        printf("Time per step: %f seconds\n", max_time / iter);
    }

    /* Clean up */
    MPI_Type_free(&column_type);
    MPI_Type_free(&row_type);
    free(T_old);
    free(T_new);

    MPI_Finalize();
    return 0;
}
