---
layout: post
title:  "Creating SSL Certificates (Linux command line)"
date:   2016-02-9 20:00:00 -0700
categories: [ "Linux" ]
---

There are so many steps to creating self signed SSL certificates, that its easy to get something wrong.  Here are some step-by-step instructions for creating certificates on a Linux command line.

Note: If you just want a CSR, try [www.digicert.com/easy-csr/openssl.htm](https://www.digicert.com/easy-csr/openssl.htm) will give you the command you need to use for OpenSSL – so easy!

First, set up some variables, replacing the values to something sensible:

```
COMPANY=Zenly
DEPARTMENT=Operations
COUNTRY_CODE=GB
STATE=London
CITY=London
AUTHORITY_NAME="Test authority"
AUTHORITY_PASSWORD=my_test_authority_password
```

Signing requests are usually sent to a signing authority. We’re going to create our own signing authority for self signed certificates.

```
openssl req -new -x509 \
   -passin pass:${AUTHORITY_PASSWORD} \
   -passout pass:${AUTHORITY_PASSWORD} \
   -extensions v3_ca -keyout ca.key -out ca.crt -days 3650 \
   -subj "/C=${COUNTRY_CODE}/ST=${STATE}/L=${CITY}/O=${COMPANY}/OU=${DEPARTMENT}/CN=${AUTHORITY_NAME}"
```

This creates:

* ca.crt – Signing authority certificate
* ca.key – Signing authority key

Each certificate that a signing authority signs needs a unique number. We’ll just create the start of this sequence:

```
echo "01" > ca.srl
```

Here’s a function we’ll use to create our certificates and sign them with our signing authority. You can just copy and paste this onto the command line if you like (no need to change anything).

```
function cert {
  COMMON_NAME=$1 # e.g. www.zenly.xyz
  FILENAME=$COMMON_NAME
  
  openssl genrsa -out ${FILENAME}.key 2048
  openssl req -new \
     -key ${FILENAME}.key \
     -out ${FILENAME}.csr \
     -subj "/C=${COUNTRY_CODE}/ST=${STATE}/L=${CITY}/O=${COMPANY}/OU=${DEPARTMENT}/CN=${COMMON_NAME}"

  openssl x509 -req -passin pass:${AUTHORITY_PASSWORD} -days 3650 \
    -in ${FILENAME}.csr -CA ca.crt -CAkey ca.key \
    -out ${FILENAME}.crt
}
```

Now let’s use the function. Here we create all the certificates for www.zenly.xyz (change the name as required). Simply repeat the command for as many self signed certificates as you need. You’ll need to have completed the steps above. i.e. set the variables at the top of this page, defined the certs function (above) and run this from within the same directory as the ca.* files we created in the first step.

```
cert www.zenly.xyz
```

This creates:

* www.zenly.xyz.key – Certificate Key
* www.zenly.xyz.csr – Signing request (to be given to the signing authority – in this case we use our own signing authority)
* www.zenly.xyz.crt – Certificate (certificate signed by our signing authority)

Using the new certificates
==========================

If we wanted to upload these to AWS, the command would be something like this:

```
aws iam upload-server-certificate \
 --server-certificate-name www.zenly.xyz \
 --certificate-body file://www.zenly.xyz.crt \
 --private-key file://www.zenly.xyz.key \
 --certificate-chain file://ca.crt
```

…or you might want to use them in Apache. Here’s how part of that configuration might look:

```
SSLEngine on
SSLCertificateFile /etc/ssl/certs/www.zenly.xyz.crt
SSLCertificateKeyFile /etc/ssl/private/www.zenly.xyz.key
SSLCertificateChainFile /etc/ssl/certs/ca.crt
```

