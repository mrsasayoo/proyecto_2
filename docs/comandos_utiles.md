#### 1. Del portátil al servidor local (Subir y borrar lo que no esté en local)
Este comando sincroniza tu portátil hacia el servidor y borra del servidor cualquier cosa que no esté en tu portátil.

```bash
rsync -avP --delete \
  --exclude="*.jpg" \
  --exclude="*.mhd" \
  --exclude="*.raw" \
  --exclude="*.zraw" \
  --exclude="*.png" \
  --exclude="*.gz" \
  --exclude="*.npy" \
  --exclude="*.zip" \
  --exclude="*.7z" \
  --exclude="*.zip.part*" \
  --exclude=".git/" \
  --exclude="venv/" \
  --exclude="__pycache__/" \
  --exclude="*.pyc" \
  /home/nicolas/Escritorio/Academico/analitica_datos/carlos_andres_ferro/proyecto_2/ \
  mrsasayo_mesa@nicolasmesa:/mnt/ssd_m2/almacenamiento/carlos_andres_ferro/proyecto_2/
```

#### 2. Del servidor local al portátil (Bajar y borrar lo que no esté en servidor)
Este comando trae todo del servidor a tu portátil y borra de tu portátil lo que no exista en el servidor.

```bash
rsync -avP --delete \
  --exclude="*.jpg" \
  --exclude="*.mhd" \
  --exclude="*.raw" \
  --exclude="*.zraw" \
  --exclude="*.png" \
  --exclude="*.gz" \
  --exclude="*.npy" \
  --exclude="*.zip" \
  --exclude="*.7z" \
  --exclude="*.zip.part*" \
  --exclude=".git/" \
  --exclude="venv/" \
  --exclude="__pycache__/" \
  --exclude="*.pyc" \
  mrsasayo_mesa@nicolasmesa:/mnt/ssd_m2/almacenamiento/carlos_andres_ferro/proyecto_2/ \
  /home/nicolas/Escritorio/Academico/analitica_datos/carlos_andres_ferro/proyecto_2/
```


## Visualización de Archivos
Ver estructura sin archivos voluminosos:
```bash
tree -h --du -C --sort=size -I "*.jpg|*.mhd|*.raw|*.zraw|*.png|*.gz|*.npy|*.zip|*.zip.part*|*.7z"
```


#### Espacio ocupado por carpetas
Resumen rápido (una línea por carpeta):
```bash
du -sh */
```

Top 10 carpetas más pesadas (recursivo):
```bash
du -ah . | sort -rh | head -n 10
```


#### Contar archivos por tipo
Scripts Python:
```bash
find . -type f -name "*.py" | wc -l
```

Imágenes DICOM (.mhd):
```bash
find . -type f -name "*.mhd" | wc -l
```

Imágenes PNG:
```bash
find . -type f -name "*.png" | wc -l
```

Archivos de datos (.npy):
```bash
find . -type f -name "*.npy" | wc -l
```

Todas las extensiones (conteo automático):
```bash
find . -type f | sed -n 's/^\..*\.\([a-zA-Z0-9]*\)$/\1/p' | sort | uniq -c | sort -rn
```


## Archivos modificados recientemente
Últimos 20 scripts Python (.py):
```bash
find . -type f -name "*.py" -printf '%TY-%Tm-%Td %TH:%TM: %p\n' | sort -r | head -n 20
```

Últimos 20 archivos (cualquier tipo):
```bash
find . -type f -printf '%TY-%Tm-%Td %TH:%TM: %p\n' | sort -r | head -n 20
```

## Limpieza
Eliminar logs
```bash
find . -type f -name "*.log" -exec rm -f {} +
```

Eliminar cachés de Python
```bash
find . -type d -name "__pycache__" -exec rm -rf {} +
```

Eliminar archivos compilados (.pyc)
```bash
find . -type f -name "*.pyc" -delete
```

## Conexión al Servidor
en nicolas@nznicolas
```bash
ssh mrsasayo_mesa@nicolasmesa
```

en mrsasayo_mesa@nicolasmesa
```bash
ssh portatil
```
