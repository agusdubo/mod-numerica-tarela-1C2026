## Guía de Ejecución

## Con Script

#### Luna
```bash
./run.sh luna --metodos euler rk2 rk4 --dt <segundos> --duracion <días> --salida <nombre_imagen>
```

#### Orión
```bash
./run.sh orion --metodo [euler | rk2 | rk4] --dt <segundos> --duracion <días>
```

## Sin Script

#### Luna
```bash
python -m graficos.trayectoria_luna --metodos euler rk2 rk4 --dt <segundos> --duracion <días> --salida <nombre_imagen>
```

#### Orión
```bash
python -m graficos.trayectoria_orion --metodo [euler | rk2 | rk4] --dt <segundos> --duracion <días>
```

## Valores Predefinidos

#### Luna
- *dt* = 60 segundos
- *duracion* = 27.3 días
- *salida* = comparacion_orbital.png

#### Orión
- *dt* = 60 segundos
- *duracion* = 10 días

## **Importante:**
- *dt* debe ser el mismo para ambos
- Para ejecutar el script *orion*, se debe ejecutar primero el de *luna*
- El escript de *luna* se puede usar con varios métodos a la vez (--metodos), mientras que el script *orion*, usa sólo 1 método a la vez (--metodo)