#!/bin/bash

##
# script simply 
# - installs postgres, psql (client) and psycopg2 (client driver). 
# - also adds a superuser, because permissioning was a bit awkward in python driver 
#
 

#
# usage
#
while getopts ":h" opt; do # n.b. getopts starts at 1st argument ($1) and stops at the 1st non-option argument
  case ${opt} in
    h)
      echo "Usage:"
      echo "    setup-db -h "
      echo "    setup-db setup [-u user]    Install postgres, postgres client and driver.
                                  Include user other than current user as a superuser. Additional configuration will be required for anyone but current user whoami."
      exit 0                                     ;;
    \?)
     echo "Invalid Option: -$OPTARG" 1>&2
     exit 1                                      ;;
  esac
done
shift $((OPTIND -1))


#
# install
#
user=$(whoami) # default
subcommand=$1; 
shift 1; # shift to ignore 1st non-option arg

case "$subcommand" in
  setup)
    
  
    # see if user is specified
    while getopts ":u:" opt; do
      case ${opt} in
        u )
          user=$OPTARG                       ;;
        \? )
          echo "Invalid Option: -$OPTARG" 1>&2
          exit 1                                  ;;
        :)
          echo "-$OPTARG requires an argument."
          exit 1                                  ;;
        
      esac
    done
  
    # finally go ahead with install... 
  
    # dependencies

   
   #linux
   
   #sudo apt update;
   #sudo apt install postgresql postgresql-contrib postgresql-client postgresql-client-common libpq-dev;

   
   #MacOS
    
   brew update
   brew install postgresql
   brew install libpq

   python3 -m pip install psycopg2; # TODO virtualenv? 
                                     # TODO rebuild http://initd.org/psycopg/articles/2018/02/08/psycopg-274-released/

   

    # user

    # postgres permissions are slightly annoying, couldn't set up with python client (which is why this script is not just a few `apt installs`)
   brew services start postgresql
   psql postgres -c"CREATE USER $user WITH PASSWORD '';" 
   psql postgres -c"ALTER USER $user WITH SUPERUSER;"

  ;;
esac
